#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   autoencoder.py
@Time    :   2023/07/24 11:07:45
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import tensorflow as tf
from tensorflow.keras import layers


class Autoencoder(tf.keras.Model):
    def __init__(self, data_type, input_shape, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.data_type = data_type
        self._input_shape = input_shape

    def build_ae(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def build(self, input_shape=None):
        super(Autoencoder, self).build(input_shape=((None,) + self._input_shape if input_shape is None else input_shape))


class TabularAutoencoder(Autoencoder):
    def __init__(self, input_dim=10, hidden_dim=10, encoding_dim=3):
        super(TabularAutoencoder, self).__init__(data_type='tabular', input_shape=(input_dim,))
        # Define the encoder and decoder layers
        self.encoder = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            layers.Dense(encoding_dim, activation='relu')
        ], name='encoder')
        self.encoding_dim = encoding_dim
        self.decoder = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(encoding_dim,)),
            layers.Dense(input_dim, activation='sigmoid')
        ], name='decoder')


class TextAutoencoder(Autoencoder):
    def __init__(self, vocab_size=10000, embedding_dim=50, sequence_length=100, encoding_dim=30):
        super(TextAutoencoder, self).__init__(data_type='text', input_shape=(sequence_length, ))
        # Define the encoder and decoder layers
        self.encoder = tf.keras.Sequential([
            layers.Embedding(vocab_size, embedding_dim, input_length=sequence_length),
            layers.LSTM(encoding_dim)
        ], name='encoder')
        # Decoder: outputs a sequence of probability distributions over the vocabulary for each position in the input sequence
        # Each distribution is represented by a vector of size vocab_size, where each element represents the probability of a specific word in the vocabulary.
        self.decoder = tf.keras.Sequential([
            layers.RepeatVector(sequence_length),
            layers.LSTM(embedding_dim, return_sequences=True),
            layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))
        ], name='decoder') 


class ImageAutoencoder(Autoencoder):
    def __init__(self, input_shape=(28, 28, 1), hidden_dim=32, encoding_dim=64, kernel_size=3, strides=2):
        super(ImageAutoencoder, self).__init__(data_type='image', input_shape=input_shape)
        # Define the encoder and decoder layers
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(hidden_dim, (kernel_size, kernel_size), activation='relu', padding='same', strides=strides),
            layers.Conv2D(encoding_dim, (kernel_size, kernel_size), activation='relu', padding='same', strides=strides)
        ], name='encoder')
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(encoding_dim, kernel_size=kernel_size, strides=strides, activation='relu', padding='same'),
            layers.Conv2DTranspose(hidden_dim, kernel_size=kernel_size, strides=strides, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')
        ], name='decoder')


class Conv1DTextAutoencoder(Autoencoder):
    def __init__(self, vocab_size=215, embedding_dim=128, sequence_length=100, kernel_size=3, filters=64, emb_layer=True):
        if emb_layer:
            encoder_layers = [layers.Embedding(vocab_size, embedding_dim)] 
            data_type ='text' 
        else:
            encoder_layers = []
            data_type ='embedding'
        super(Conv1DTextAutoencoder, self).__init__(data_type=data_type, input_shape=(sequence_length, ))
        # Define the encoder
        encoder_layers += [
            layers.Conv1D(filters, kernel_size, activation='relu', padding='same'),
            layers.GlobalMaxPooling1D()
        ]
        if not emb_layer:
            encoder_layers += [
                layers.Dense(filters, activation='relu')
            ]
        self.encoder = tf.keras.Sequential(encoder_layers, name='encoder')
        # Define the decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(sequence_length, activation='relu'),
            layers.Reshape((sequence_length, 1)),
            layers.Conv1DTranspose(embedding_dim, kernel_size, activation='relu', padding='same'),
            # layers.Conv1D(vocab_size, kernel_size, activation='softmax', padding='same')
        ], name='decoder')


class SeqAutoencoder(Autoencoder):
    def __init__(self, input_shape=(100, 10), filters=32, kernel_size=3, pool_size=2):
        super(SeqAutoencoder, self).__init__(data_type='sequence', input_shape=input_shape)
        # Define the encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(filters, kernel_size, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size)
        ], name='encoder')
        # Define the decoder
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(filters, kernel_size, strides=pool_size, activation='relu', padding='same'),
            layers.Conv1D(input_shape[1], kernel_size, activation='sigmoid', padding='same')
        ], name='decoder')


if __name__ == "__main__":
    autoencoder = TextAutoencoder(vocab_size=10000, embedding_dim=50, sequence_length=100, encoding_dim=30)
