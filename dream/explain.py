#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cade_explain.py
@Time    :   2024/01/10 13:44:23
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2024
@Desc    :   Adapted from CADE, support tf2
'''

# here put the import lib
from numpy.random import seed
import random
random.seed(1)
seed(1)

import tensorflow as tf
import logging
import numpy as np

import warnings
warnings.filterwarnings('ignore')
from dream.concept_learning import concept_presence_loss


def explain_instance(x, dream_model, diff_idx, centroid, closest_to_centroid_sample,
                     distance_lowerbound, lambda_1, cert_loss=True, concept_space=False, closest_pred=None, yb_centroid=None, **kwargs):
    LR = 1e-2  # learning rate
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)
    EXP_EPOCH = 250
    EXP_LAMBDA_PATIENCE = 20
    EARLY_STOP_PATIENCE = 10
    USE_GUMBLE_TRICK = True

    TEMP = 0.1
    M1 = np.zeros_like(x, dtype=np.float32)
    M1[diff_idx] = 1

    if concept_space:
        x = dream_model.encoder.predict(x[None,:])[0]
        closest_to_centroid_sample = dream_model.encoder.predict(closest_to_centroid_sample[None,:])[0]
        explicit_flg = kwargs.get('explicit', False)
        if explicit_flg:
            M1[dream_model.num_concept+1:] = 0
        else:
            M1 = np.ones_like(x)
    else:
        x = x.astype(np.float32)
        closest_to_centroid_sample = closest_to_centroid_sample.astype(np.float32)
    
    MASK_SHAPE = x.shape
    logging.debug(f'MASK_SHAPE: {MASK_SHAPE}')
    logging.debug(f'distance lowerbound: {distance_lowerbound}')
    logging.debug(f'epoch: {EXP_EPOCH}')
    logging.debug(f'temperature: {TEMP}')
    logging.debug(f'use gumble trick: {USE_GUMBLE_TRICK}')

    mask_best = None

    exp_test = DriftExplainer(
        batch_size=10,
        mask_shape=MASK_SHAPE,
        model=dream_model,
        optimizer=OPTIMIZER,
        temp=TEMP,
        cert_loss=cert_loss,
        concept_space=concept_space)

    mask_best = exp_test.fit_local(
        x_t=x,
        b=M1, # b[i]=0: x_t[i] == x_c[i]
        c_y=centroid,
        x_c=closest_to_centroid_sample,
        num_sync=50,
        num_changed_fea=1,
        epochs=EXP_EPOCH,
        lambda_1=lambda_1,
        y_c_pred=closest_pred, 
        yb_c=yb_centroid,
        exp_loss_lowerbound=distance_lowerbound,
        lambda_patience=EXP_LAMBDA_PATIENCE,
        early_stop_patience=EARLY_STOP_PATIENCE)

    logging.debug(f'M1 * mask == 1: {np.where(M1 * mask_best == 1)[0]}')

    if mask_best is not None:
        return M1 * mask_best


class DriftExplainer(tf.Module):
    def __init__(self, batch_size, mask_shape, model, optimizer, temp=0.1, cert_loss=False, concept_space=False):
        self.batch_size = batch_size
        self.mask_shape = mask_shape
        self.model = model
        self.temp = temp
        self.cert_loss = cert_loss
        self.concept_space = concept_space 
        self.model.concept_space = concept_space
        self.optimizer = optimizer
        self.p = tf.Variable(tf.random.uniform(shape=mask_shape, minval=0, maxval=1), trainable=True)

    def sample_m(self, p_normalized, batch_size, temp=1.0 / 10.0):
        """Sample m(approximated binary output) using the current probabilities p."""
        epsilon = np.finfo(float).eps  # 1e-16
        uniform_random = tf.random.uniform(shape=(batch_size, )+self.mask_shape, minval=0, maxval=1)
        reverse_theta = 1 - p_normalized
        reverse_unif_noise = 1 - uniform_random
        appro = tf.math.log(p_normalized + epsilon) - tf.math.log(reverse_theta + epsilon) + \
                tf.math.log(uniform_random) - tf.math.log(reverse_unif_noise)
        logit = appro / temp
        return tf.sigmoid(logit)

    def compute_loss(self, x_t, x_c, m, b, c_y, **kwargs):
        """Compute the loss function as described in the paper."""
        # Perturbed sample z_t
        x_prime = x_t * (1 - m * b) + x_c * (m * b)
        z_t = self.model.encoder(x_prime, training=False) \
            if not self.concept_space else x_prime
        # Latent space distance term
        distance_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(z_t - c_y), axis=1)))
        # Regularization term
        regularization_loss = tf.reduce_sum(tf.abs(m * b)) + tf.sqrt(tf.reduce_sum(tf.square(m * b)))
        if self.cert_loss:
            uncertainty_loss = self.model.scoring_fn(x_prime, reduction='auto', reference_pred=kwargs.get('y_c_pred'))
            yb_c = kwargs.get('yb_c', None)
            if self.concept_space and yb_c is not None:
                concept_loss = concept_presence_loss(self.model.concept_dense(z_t[:, :self.model.num_concept]), yb_c)
            else: concept_loss = 0.
            return (distance_loss, uncertainty_loss, concept_loss), regularization_loss
        # Total loss
        return distance_loss, regularization_loss

    @tf.function
    def train_step(self, X_t, x_c, b, c_y, lambda_1, **kwargs):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            p_normalized = tf.clip_by_value(self.p, 0.0, 1.0)
            m = self.sample_m(p_normalized, tf.shape(X_t)[0], temp=self.temp)
            loss_exp, loss_reg = self.compute_loss(X_t, x_c, m, b, c_y, **kwargs)
            if self.cert_loss:
                loss_sep, loss_unc, loss_con = loss_exp
                loss = loss_sep + lambda_1 * loss_reg + kwargs.get('lambda_unc', 1.) * loss_unc + kwargs.get('lambda_con', 10.) * loss_con
            else:
                loss = loss_exp + lambda_1 * loss_reg
        grads = tape.gradient(loss, [self.p])
        self.optimizer.apply_gradients(zip(grads, [self.p]))
        return loss, loss_exp, loss_reg, p_normalized

    def fit_local(self, x_t, b, c_y, x_c, num_sync, num_changed_fea, epochs, lambda_1, y_c_pred=None, yb_c=None,exp_loss_lowerbound=0.17, iteration_threshold=1e-4, lambda_patience=100, lambda_multiplier=1.5, early_stop_patience=10, **kwargs):
        """Fit the model to the data."""
        # Prepare synthesized samples
        sync_idx = np.random.choice(x_t.shape[0], (num_sync, num_changed_fea))
        sync_x = np.repeat(x_t[None, :], num_sync, axis=0).reshape(num_sync, x_t.shape[0])
        for i in range(num_sync):
            if self.concept_space:
                sync_x[i, sync_idx[i]] = x_t[sync_idx[i]] + np.random.uniform(low=-0.5, high=0.5)
            else:
                sync_x[i, sync_idx[i]] = 1 - x_t[sync_idx[i]]

        input_ = np.vstack((x_t, sync_x))
        # Prepare for training
        num_batches = int(np.ceil(input_.shape[0] / self.batch_size))
        idx = np.arange(input_.shape[0])        
        
        # Initialize tracking variables
        loss_best = float('inf')
        loss_sparse_mask_best = float('inf')
        loss_last = float('inf')
        loss_sparse_mask_last = float('inf')

        mask_best = None
        early_stop_counter = 0
        lambda_up_counter = 0
        lambda_down_counter = 0
        
        for epoch in range(epochs):
            np.random.shuffle(idx)
            loss_tmp = []
            loss_exp_tmp = []
            loss_sparse_mask_tmp = []
            for i in range(num_batches):
                batch_idx = idx[i * self.batch_size:min((i + 1) * self.batch_size, input_.shape[0])]
                input_batch = input_[batch_idx, :]
                if y_c_pred is not None:
                    _y_c_pred = np.tile(y_c_pred, [len(input_batch), 1])
                if yb_c is not None:
                    _yb_c = np.tile(yb_c, [len(input_batch), 1])
                loss, loss_exp, loss_reg, p_normalized = self.train_step(input_batch, x_c, b, c_y, lambda_1, y_c_pred=_y_c_pred,yb_c=_yb_c, **kwargs)
                if self.cert_loss:
                    # print([l.numpy() for l in loss_exp])
                    loss_exp = loss_exp[0]
                loss_tmp.append(loss.numpy())
                loss_exp_tmp.append(loss_exp.numpy())
                loss_sparse_mask_tmp.append(loss_reg.numpy())

            loss = np.mean(loss_tmp)
            loss_exp = np.mean(loss_exp_tmp)
            loss_sparse_mask = np.mean(loss_sparse_mask_tmp)

            # Adjust lambda_1 based on loss_exp
            if loss_exp <= exp_loss_lowerbound:
                lambda_up_counter += 1
                if lambda_up_counter >= lambda_patience:
                    lambda_1 = lambda_1 * lambda_multiplier
                    lambda_up_counter = 0
            else:
                lambda_down_counter += 1
                if lambda_down_counter >= lambda_patience:
                    lambda_1 = lambda_1 / lambda_multiplier
                    lambda_down_counter = 0

            # Early stopping check
            if (np.abs(loss - loss_last) < iteration_threshold) or \
                        (np.abs(loss_sparse_mask - loss_sparse_mask_last) < iteration_threshold):
                early_stop_counter += 1

            if (loss_exp <= exp_loss_lowerbound) and (early_stop_counter >= early_stop_patience):
                logging.debug('Reach the threshold and stop training at iteration %d/%d.' % (epoch + 1, epochs))
                mask_best = p_normalized
                break
            
            # Update best loss
            if loss_best > loss or loss_sparse_mask_best > loss_sparse_mask:
                logging.debug("Epoch %d/%d: loss = %.5f explanation_loss = %.5f "
                                        "mask_sparse_loss = %.5f "
                                        % (epoch+1, epochs, loss, loss_exp, loss_sparse_mask))
                loss_best = loss
                loss_sparse_mask_best = loss_sparse_mask
                mask_best = p_normalized
            
            loss_last = loss
            loss_sparse_mask_last = loss_sparse_mask

        if mask_best is None:
            logging.info(f'did NOT find the best mask')
        
        return mask_best.numpy()
    