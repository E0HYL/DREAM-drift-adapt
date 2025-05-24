#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_utils.py
@Time    :   2023/07/26 15:05:44
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
import logging
import copy
import json
import inspect
import numpy as np
from glob import glob
import tensorflow as tf


def train_model(X_train, y_train, basemodel, model_path=None, loss='categorical_crossentropy', batch_size=32, num_epochs=30, metrics=['accuracy'], mon_metric=None, mode=None, lr=0.001, debug=False, alternate_monitor='loss', verbose=1, verbose_batch=True, **kwargs):
    monitor = metrics[0] if mon_metric is None else mon_metric
    mode = mode if mode is not None else ('min' if 'loss' in monitor else 'max')
    callbacks = create_checkpoint(model_path, monitor=monitor, alternate_monitor=alternate_monitor, verbose=verbose, verbose_batch=verbose_batch, mode=mode)
    basemodel = default_classifier_compile(basemodel, loss, metrics, lr, debug)
    if num_epochs > 0:
        if y_train is not None:
            basemodel.fit(X_train, y_train, batch_size=batch_size, callbacks=callbacks, epochs=num_epochs, **kwargs)
        else:
            basemodel.fit(X_train, batch_size=batch_size, callbacks=callbacks, epochs=num_epochs, **kwargs) # dataset
    if model_path is not None:
        basemodel.load_weights(model_path) # the best model after training
    return basemodel


def create_checkpoint(model_path, monitor='accuracy', alternate_monitor=None, verbose=1, verbose_batch=True, save_best_only=True, save_weights_only=True, mode='max'):
    if model_path is not None:
        mcp_save = tf.keras.callbacks.ModelCheckpoint(model_path,
                                monitor=monitor,
                                save_best_only=save_best_only,
                                save_weights_only=save_weights_only,
                                verbose=verbose,
                                mode=mode) \
                                if alternate_monitor is None else \
                    CustomModelCheckpoint(alternate_monitor,
                                model_path,
                                monitor=monitor,
                                save_best_only=save_best_only,
                                save_weights_only=save_weights_only,
                                verbose=verbose,
                                mode=mode)
        callbacks = [mcp_save] if verbose and verbose_batch else [CustomProgbarLogger(count_mode='steps'), mcp_save]
    else:
        callbacks = None
    return callbacks


def default_classifier_compile(model, loss='categorical_crossentropy', metrics=['accuracy'], lr=0.001, debug=False, **kwargs):
    if not isinstance(lr, dict):
        lr = {'optimizer': tf.keras.optimizers.Adam(learning_rate=lr)} # model.optimizer,
    model.compile(loss=loss,
                  metrics=metrics, 
                  run_eagerly=debug,
                  **lr, **kwargs)
    return model


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, alternate_monitor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_monitor = self.monitor  # Save the original metric to monitor
        self.value_limit = .99 if 'acc' in self._original_monitor else None
        self._alternate_monitor = alternate_monitor  # Replace with the metric you'd like to monitor after accuracy reaches 1.0

    def on_epoch_end(self, epoch, logs=None):
        best = self.best
        super().on_epoch_end(epoch, logs)
        if self.monitor_op(logs.get(self.monitor), best): 
            logging.info(format_epoch_log(epoch, logs))
        if self.value_limit is not None:
            if self.monitor == self._original_monitor and (self.monitor_op(logs.get(self._original_monitor), self.value_limit) or logs.get(self._original_monitor) == self.value_limit):
                self.monitor = self._alternate_monitor  # Switch to alternate metric
                self.best = logs.get(self.monitor)
                print(f"{self._original_monitor} reached {self.value_limit}. Switching to monitor {self.monitor}.")   
                if 'loss' in self.monitor:
                    self.monitor_op = np.less


def format_epoch_log(epoch, logs):
    format_str = f'Epoch {epoch+1}: '
    for k, v in logs.items():
        if 'categorical_accuracy' in k:
            k = 'accuracy'
        format_str += f'{k}@{format_loss_value(v)}, '
    format_str = format_str.strip(', ')
    return format_str


class CustomProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, *args, **kwargs):
        self.simplify_epoch_end = kwargs.pop('simplify_epoch_end', False)
        super().__init__(*args, **kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.original_logs = None

    def on_train_batch_end(self, batch, logs=None):
        logs = self.simplify_logs(logs)
        # Now call the original on_batch_end method to display the modified logs
        super().on_train_batch_end(batch, logs)
        self.restore_logs(logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.simplify_epoch_end:
            simple_logs = self.simplify_logs(logs)
            super().on_epoch_end(epoch, simple_logs)
            self.restore_logs(logs, finalize=True)
        else:
            super().on_epoch_end(epoch, logs)

    def simplify_logs(self, logs):
        # Create a shallow copy of the original logs
        if self.original_logs is None:
            self.original_logs = copy.deepcopy(logs)
        # Modify the logs dictionary as per your requirements
        keys_to_delete = [key for key in logs if '_loss' in key]
        for key in keys_to_delete:
            del logs[key]
        return logs
    
    def restore_logs(self, logs, finalize=False):
        # Restore the original logs so as not to affect other callbacks
        if self.original_logs is not None:
            logs.clear()
            logs.update(self.original_logs)
        if not finalize:    
            self.original_logs = None


def adjust_ind_labels(labels):
    unique_values = []
    for label in labels:
        unique_values += list(np.unique(label))
    depth = len(set(unique_values))
    return [tf.one_hot(label, depth) for label in labels]


def create_contrastive_batches(X_train, y_train, batch_size, similar_samples_ratio=0.25, yb_train=None, infinite=False):
    flag = True
    while flag:
        if batch_size % 4 != 0:
            raise ValueError('batch_size should be a multiple of 4.')

        half_size = int(batch_size / 2)
        num_sim = int(batch_size * similar_samples_ratio)
        random_idx = np.random.permutation(X_train.shape[0]) 

        # For similar and dissimilar samples
        index_cls, index_no_cls = [], []
        for label in range(len(np.unique(y_train))):
            index_cls.append(np.where(y_train == label)[0])
            index_no_cls.append(np.where(y_train != label)[0])

        num_batches = X_train.shape[0] // batch_size

        for b in range(num_batches):
            # Prepare the batch data
            b_X_train = X_train[random_idx[b * half_size: (b + 1) * half_size]]
            b_y_train = y_train[random_idx[b * half_size: (b + 1) * half_size]]
            if yb_train is not None: 
                b_yb_train = yb_train[random_idx[b * half_size: (b + 1) * half_size]]

            # Fill the rest of the batch with similar/dissimilar samples
            for m in range(num_sim):
                pair = np.random.choice(index_cls[b_y_train[m]], 1)
                b_X_train = np.append(b_X_train, [X_train[pair[0]]], axis=0)
                b_y_train = np.append(b_y_train, [y_train[pair[0]]])
                if yb_train is not None:
                    b_yb_train = np.append(b_yb_train, [yb_train[pair[0]]], axis=0)
            for m in range(num_sim, half_size):
                pair = np.random.choice(index_no_cls[b_y_train[m]], 1)
                b_X_train = np.append(b_X_train, [X_train[pair[0]]], axis=0)
                b_y_train = np.append(b_y_train, [y_train[pair[0]]])
                if yb_train is not None:
                    b_yb_train = np.append(b_yb_train, [yb_train[pair[0]]], axis=0)
            
            if yb_train is None:
                yield b_X_train, b_y_train
            else:
                yield b_X_train, b_y_train, b_yb_train
        flag = infinite


def format_loss_value(loss_value):
    if loss_value < 1e-3:
        return np.format_float_scientific(loss_value, exp_digits=2, unique=False, precision=4)
    else:
        return '%.4f' % loss_value


def generate_contrast_dataset(x_train, y_train, batch_size, input_shape, yb_train=None, infinite=False):
    gen = create_contrastive_batches(x_train, y_train, batch_size, yb_train=yb_train, infinite=infinite)
    dataset_shape = (
            tf.TensorSpec(shape=(None, )+input_shape, dtype=tf.as_dtype(x_train.dtype)),  # dynamically assign the data type of x_train
            tf.TensorSpec(shape=(None,), dtype=tf.as_dtype(y_train.dtype))  # dynamically assign the data type of y_train
        )
    if yb_train is not None:
        dataset_shape = ( *dataset_shape, tf.TensorSpec(shape=(None, )+yb_train.shape[1:], dtype=tf.as_dtype(yb_train.dtype)) )
    dataset = tf.data.Dataset.from_generator(
        lambda: gen, 
        output_signature=dataset_shape
    ) # output_signature: corresponding to the structure of the elements that will be generated by the generator (x and y).
    return dataset


# Create contrastive batches for each epoch; Fix the bug for training model with customized loss (tf2.8.0); monitor and log model saving state manually
def train_custom_model(model, x_train, y_train, batch_size, input_shape, epochs, base_value=np.inf, lr=0.0001, monitor='loss', mode='min', model_path=None, **kwargs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), run_eagerly=True) 
    min_loss = base_value
    for epoch in range(epochs):
        dataset = generate_contrast_dataset(x_train, y_train, batch_size, input_shape)
        # Train the model for one epoch
        history = model.fit(dataset, epochs=1, verbose=2, **kwargs) # bug for verbose=1
        current_loss = history.history.pop(monitor)[0]
        exceed_flag = current_loss < min_loss if mode=='min' else current_loss > min_loss
        if model_path is not None and exceed_flag:
            logging.info(f"Epoch {epoch}: Loss improved to {format_loss_value(current_loss)} {[f'{k}@{format_loss_value(v[0])}' for k, v in history.history.items()]}, saving model.")
            min_loss = current_loss
            model.save_weights(model_path)


class ModelConfig(object):
    def __init__(self, config_path=None):
        self.config_path = config_path

    def _set_config_path(self, data_name, feature_name, model_name):
        self.config_path = os.path.join('models', data_name, feature_name.capitalize(), model_name, 'model_config.txt')

    def get_config(self, data_name, feature_name, model_name):
        self._set_config_path(data_name, feature_name, model_name)
        return self.load_config()
    
    def save_config(self, config):
        with open(self.config_path, 'w') as fp:
            json.dump(config, fp)

    def load_config(self):
        config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as fp:
                config = json.load(fp)
        return config


def classification_loss(classifier_labels, classifier_predictions, reduction='auto'):
    if len(classifier_labels.shape) == 1:
        return tf.keras.losses.SparseCategoricalCrossentropy(reduction=reduction)(classifier_labels, classifier_predictions)
    else:
        return tf.keras.losses.CategoricalCrossentropy(reduction=reduction)(classifier_labels, classifier_predictions)


def get_cls_uncertainty(y_pred, entropy=True):
    return -np.sum(y_pred * np.log(y_pred + 1e-10), axis=1) if entropy else 1 - (y_pred).max(-1)


def get_cls_pseudo_loss(y_pred):
    pesudo_label = np.argmax(y_pred, axis=1) # softmax output
    pseudo_loss = classification_loss(pesudo_label, y_pred, reduction='none')
    return pseudo_loss


def call_accepts_training(layer):
    sig = inspect.signature(layer.call)
    return 'training' in sig.parameters


def rename_models(d):
    fs = glob(os.path.join(d, '*_*.h5'))
    print(fs)
    for f in fs:
            x = f.split('/')[-1]
            _ori = os.path.join(d, x[0]+'.h5')
            if os.path.exists(_ori):
                    os.rename(_ori, _ori+'.bp')
            os.rename(f, _ori)


def clear_models():
    tf.keras.backend.clear_session()


def form_tf_data(data, batch_size=32, shuffle=True, **shuffle_args):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        buffer_size = shuffle_args.get('buffer_size', 11000)
        # whether the shuffle order should be different for each epoch
        reshuffle_each_iteration = shuffle_args.get('reshuffle_each_iteration', True) 
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def binary_crossentropy(y_true, y_pred, from_logits=False, pos_weight=1.):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    if from_logits:
        # Compute binary cross-entropy from logits
        # Logits are the outputs of a model's final layer without a sigmoid activation function applied.
        # loss = tf.math.maximum(y_pred, 0) - y_pred * y_true + tf.math.log(1 + tf.math.exp(-tf.math.abs(y_pred)))
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight) # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    else:  
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # Compute binary cross-entropy from probabilities
        loss = - (y_true * tf.math.log(y_pred) * pos_weight + (1 - y_true) * tf.math.log(1 - y_pred))
    return loss


def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=1., label_smoothing=0.0, reduction='auto', sample_weight=None):
    if label_smoothing > 0:
        smooth_positives = 1.0 - label_smoothing
        smooth_negatives = label_smoothing / tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * smooth_positives + smooth_negatives

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.keras.backend.log(y_pred)
    # Calculate Focal Loss
    loss = alpha * tf.keras.backend.pow(1 - y_pred, gamma) * cross_entropy
    if sample_weight is not None:
        sample_weight = tf.reshape(tf.cast(sample_weight, loss.dtype), (-1, 1))  # Reshape to (batch_size, 1)
        loss = loss * sample_weight
    
    if reduction == 'sum':
        return tf.keras.backend.sum(loss)
    elif reduction == 'none':
        return tf.keras.backend.sum(loss, axis=-1)
    else:  # 'auto' or 'mean'
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
    

def get_pesudo_label(y_pred=None, y_concept_pred=None, **kwargs):
    if y_concept_pred is None:
        assert y_pred is not None
        return np.argmax(y_pred, axis=1)
    concept_label_flag = kwargs.get('concept_label_flag', True)
    pseudo_concept_label = np.where(y_concept_pred > .5, 1., 0.)
    pseudo_label = tf.sign(tf.reduce_sum(pseudo_concept_label, axis=-1))
    if concept_label_flag:
        return pseudo_label, pseudo_concept_label
    return pseudo_label


def get_model_size(model, bytes_per_param=4):
    total_params = 0
    for variable in model.trainable_variables:
        total_params += tf.size(variable).numpy()
    model_size_bytes = total_params * bytes_per_param
    model_size_mb = model_size_bytes / 1e6
    return model_size_mb
