#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   baseline_cade.py
@Time    :   2023/07/24 12:24:29
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dream.train_utils import train_custom_model


class ContrastiveAutoencoder(tf.keras.Model):
    def __init__(self, autoencoder, margin=10.0, lambda_=1e-1):
        super(ContrastiveAutoencoder, self).__init__()
        self.autoencoder = autoencoder
        self.margin = margin
        self.lambda_ = lambda_

    def call(self, inputs):
        decoded, encoded = self.autoencoder(inputs)
        return decoded, encoded

    def train_step(self, data):
        # Unpack the data. Note: The `data` argument comes from the `tf.data.Dataset`.
        x, y = data
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, contrastive_loss = self.calculate_losses(x, y)

        # Compute gradients
        trainable_vars = self.autoencoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return a dict mapping metric names to current value. Needed for Keras to display losses.
        return {'loss': total_loss, 'reconstruction_loss': reconstruction_loss, 'contrastive_loss': contrastive_loss}

    def calculate_losses(self, x, y):
        # Create positive and negative pairs
        half_batch_size = tf.shape(x)[0] // 2
        left_p = tf.range(0, half_batch_size, dtype=tf.int32)
        right_p = tf.range(half_batch_size, 2 * half_batch_size, dtype=tf.int32)
        is_same = tf.cast(tf.equal(tf.gather(y, left_p), tf.gather(y, right_p)), tf.float32)
        # Forward pass
        reconstructed, encoded = self.autoencoder(x)
        # Compute the reconstruction loss
        if self.autoencoder.data_type == 'text':
            x = self.autoencoder.encoder.layers[0](x)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, reconstructed))

        # Compute the contrastive loss
        axis = [i+1 for i in range(len(encoded.shape)-1)]
        dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.gather(encoded, left_p), tf.gather(encoded, right_p))), axis=axis) + 1e-10)
        contrastive_loss = is_same * dist + (1.0 - is_same) * tf.nn.relu(self.margin - dist)
        contrastive_loss = tf.reduce_mean(contrastive_loss)

        # Compute the total loss
        total_loss = reconstruction_loss + contrastive_loss * self.lambda_
        return total_loss, reconstruction_loss, contrastive_loss


def epoch_train(x_train, y_train, epochs, batch_size, input_shape, autoencoder, _model_path, save_bp=False, lr=.0004, continue_train=False, **kwargs):
    # Instantiate the model
    model = ContrastiveAutoencoder(autoencoder, **kwargs) # lambda_, margin
    model_path = _model_path.replace('.h5', f'_lambda-{model.lambda_}_margin-{model.margin}.h5') if save_bp else _model_path
    def _load_existing_weights(model_path=model_path):
        # model.built = True
        model.build((None,) + input_shape)
        if os.path.exists(model_path):
            logging.info(f'loading model weights from {model_path}...')
            model.load_weights(model_path)
            return True
        return False

    if epochs > 0:
        logging.info(f"Training CADE [lamda={model.lambda_}, margin={model.margin}] for {epochs} epochs. Model weights will be saved at '{model_path}'.")
        if continue_train: 
            _load_existing_weights(_model_path)
            min_loss = model.evaluate(x_train)[0]
        else:
            min_loss = np.inf
        train_custom_model(model, x_train, y_train, batch_size, input_shape, epochs, min_loss, lr, model_path=model_path)
        if save_bp: model.save_weights(_model_path)
    
    assert _load_existing_weights(model_path), f'Trained model does not exist at {model_path}'
    best_model = model
    return best_model


def detect_drift(x_train, y_train, x_test, ae_model, mad_threshold=3.5, batch_size=32):
    z_train = ae_model.encoder.predict(x_train, batch_size=batch_size)
    z_test = ae_model.encoder.predict(x_test, batch_size=batch_size)

    N, N_family, z_family = get_latent_data_for_each_family(z_train, y_train)
    centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
    dis_family = get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family)
    # Median Absolute Deviation (MAD): setting the distance threshold for each class
    mad_family = get_MAD_for_each_family(dis_family, N, N_family)

    anomaly_scores = []
    for z_k in tqdm(z_test, desc='Drift Detection'):
        dis_k = [np.linalg.norm(z_k - centroids[i]) for i in range(N)]
        anomaly_k = [np.abs(dis_k[i] - np.median(dis_family[i])) / mad_family[i] for i in range(N)]
        min_anomaly_score = np.min(anomaly_k)
        anomaly_scores.append(min_anomaly_score)
    is_drift = [True if value > mad_threshold else False for value in anomaly_scores]
    return anomaly_scores, is_drift


def get_latent_data_for_each_family(z_train, y_train):
    N = len(np.unique(y_train))
    N_family = [len(np.where(y_train == family)[0]) for family in range(N)]
    z_family = []
    for family in range(N):
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)

    # z_len = [len(z_family[i]) for i in range(N)]
    # logging.debug(z_len)
    return N, N_family, z_family


def get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family):
    dis_family = []  # two-dimension list

    for i in range(N): # i: family index
        dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
        dis_family.append(dis)

    # dis_len = [len(dis_family[i]) for i in range(N)]
    # logging.debug(dis_len)
    return dis_family


def get_MAD_for_each_family(dis_family, N, N_family):
    mad_family = []
    for i in range(N):
        median = np.median(dis_family[i])
        # logging.debug(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median) for j in range(N_family[i])]
        median_diff = np.maximum(np.median(diff_list), np.finfo(type(median)).eps) # prevent zero values
        mad = 1.4826 * median_diff  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    logging.debug(f'mad_family: {mad_family}')

    return mad_family


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # set your parameters
    dims = [10, 5, 2]
    batch_size = 16
    margin = 1.0
    lambda_ = 1.0
    epochs = 10

    # Create some random numerical data and labels
    x_train = np.random.uniform(size=(1000, dims[0]))
    y_train = np.random.randint(0, 3, size=(1000,))

    from autoencoder import TabularAutoencoder
    autoencoder = TabularAutoencoder(input_dim=dims[0], hidden_dim=dims[1], encoding_dim=dims[2])
    epoch_train(x_train, y_train, epochs=epochs, batch_size=batch_size, input_shape=(dims[0],), autoencoder=autoencoder, lambda_=lambda_, margin=margin)