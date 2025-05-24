#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ood_detector.py
@Time    :   2023/12/18 13:17:37
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
import tensorflow as tf
from dream.concept_learning import concept_separation_loss, reconstruction_loss
from dream.train_utils import classification_loss
from dream.baseline_cade import detect_drift
from dream.evaluate import draw_roc_curve


class ContrastiveDetector(tf.keras.Model): 
    # unified implementations for CADE and HCC
    def __init__(self, autoencoder, margin=10.0, lambda_=1e-1, hierarchical=False):
        super(ContrastiveDetector, self).__init__()
        self.autoencoder = autoencoder
        self.margin = margin
        self.lambda_ = lambda_
        self.hierarchical = hierarchical

    def call(self, inputs):
        decoded, encoded = self.autoencoder(inputs)
        return decoded, encoded

    def train_step(self, data):
        # Unpack the data. Note: The `data` argument comes from the `tf.data.Dataset`.
        x, y = data
        with tf.GradientTape() as tape:
            total_loss, main_loss, contrastive_loss = self.calculate_losses(x, y)

        # Compute gradients
        trainable_vars = self.autoencoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return a dict mapping metric names to current value. Needed for Keras to display losses.
        return {'total_loss': total_loss, self.main_loss: main_loss, 'contrastive_loss': contrastive_loss}

    def calculate_losses(self, x, y, training=True, classifier=None, **kwargs):
        reconstructed, encoded = self.autoencoder(x)
        seperation_loss_metric = concept_separation_loss(encoded, y, self.margin, self.hierarchical, training)
        if self.hierarchical:
            y_pred = reconstructed if classifier is None else classifier.predict(x)
            # cross entropy of mlp classifier decoder
            self.main_loss = 'classification_loss'
            binary_labels = tf.cast(tf.not_equal(y, 0), tf.int32)
            if training or kwargs.get('nb_unc', False): # NCE for testing
                balance_loss_metric = classification_loss(binary_labels, y_pred)
            else:  # original paper
                balance_loss_metric = classification_loss(binary_labels, y_pred, reduction='none')[0]
        else:
            # reconstruction loss
            self.main_loss = 'reconstruction_loss'
            balance_loss_metric = reconstruction_loss(x, reconstructed)
        total_loss = balance_loss_metric + self.lambda_ * seperation_loss_metric
        return total_loss, balance_loss_metric, seperation_loss_metric
    
    def detect_drift(self, X_cal, y_cal, X_test, bsize=64, classifier=None, **kwargs):
        if self.hierarchical: # pseudo loss
            y_pred, embedding = self.autoencoder(X_test)
            y_pred = y_pred if classifier is None else classifier.predict(X_test)
            pesudo_label = np.argmax(y_pred, axis=1)
            embedding_cal = self.autoencoder(X_cal)[1]
            # select the 2N-1 nearest neighbors
            # build the KDTree
            tree = KDTree(embedding_cal)
            # query all z_test up to a margin
            all_neighbors = tree.query(embedding, k=embedding_cal.shape[0], workers=8)
            all_distances, all_indices = all_neighbors
            num_test = embedding.shape[0]
            anomaly_scores = []
            n_neighbors = kwargs.get('num_near_neighbors', None)
            f_neighbors = kwargs.get('num_far_neighbors', None)
            if n_neighbors is not None or f_neighbors is not None:
                selected_neighbors = []
            for i in tqdm(range(num_test), desc='HCC Sampler'):
                test_sample = X_test[i:i+1]
                batch_indices = all_indices[i][:bsize-1]
                x_train_batch = X_cal[batch_indices] 
                if n_neighbors is not None:
                    selected_neighbors.append(batch_indices[:n_neighbors])
                if f_neighbors is not None:
                    selected_neighbors.append(all_indices[i][-f_neighbors:])
                x_batch = tf.concat((test_sample, x_train_batch), 0)
                # y_batch
                y_train_batch = y_cal[batch_indices]
                y_train_batch = tf.cast(tf.not_equal(y_train_batch, 0), tf.int32)
                y_batch = np.hstack((pesudo_label[i], y_train_batch))
                pseudo_loss = self.calculate_losses(x_batch, y_batch, False, classifier) 
                anomaly_scores.append(pseudo_loss)   
            if n_neighbors is not None or f_neighbors is not None:
                return anomaly_scores, selected_neighbors   
        else:
            anomaly_scores, _ = detect_drift(X_cal, y_cal, X_test, self.autoencoder)
        return anomaly_scores


class PseudoBatch:
    def __init__(self, autoencoder, classifier) -> None:
        self.autoencoder = autoencoder
        self.classifier = classifier

    def setup_data(self, X_test, X_cal, y_cal, batch_size):
        self.X_test = X_test
        self.X_cal = X_cal
        self.y_cal = y_cal
        y_pred, embedding = self.autoencoder.predict(X_test, batch_size=batch_size)
        y_pred = self.classifier.predict(X_test, batch_size=batch_size)
        self.pesudo_label = np.argmax(y_pred, axis=1)
        embedding_cal = self.autoencoder.predict(X_cal, batch_size=batch_size)[1]
        # build the KDTree
        tree = KDTree(embedding_cal)
        # query all z_test up to a margin
        all_neighbors = tree.query(embedding, k=embedding_cal.shape[0], workers=8)
        _, self.all_indices = all_neighbors

    def form_hcc_pseudo_batch(self, i, batch_size):
        test_sample = self.X_test[i:i+1]
        batch_indices = self.all_indices[i][:batch_size-1]
        x_train_batch = self.X_cal[batch_indices] 
        x_batch = tf.concat((test_sample, x_train_batch), 0)
        # y_batch
        y_train_batch = self.y_cal[batch_indices]
        y_train_batch = tf.cast(tf.not_equal(y_train_batch, 0), tf.int32)
        y_batch = np.hstack((self.pesudo_label[i], y_train_batch))
        return x_batch, y_batch
    

def evaluate_detector_with_model_predictions(y_pred, y_true, anomaly_scores):
    predicted_labels = tf.argmax(y_pred, axis=1)
    y_true = (y_true>0).astype(int)
    correct_predictions = tf.equal(predicted_labels, y_true)
    wrong_predictions = tf.cast(tf.logical_not(correct_predictions), tf.int32)
    anomaly_scores = np.array(anomaly_scores)
    if len(anomaly_scores.shape) == 1:
        roc = draw_roc_curve(wrong_predictions, anomaly_scores)
    if len(anomaly_scores.shape) == 2:
        roc = []
        for score_index in range(anomaly_scores.shape[1]):
            score = anomaly_scores[:, score_index]
            roc.append(draw_roc_curve(wrong_predictions, score))
    return roc
