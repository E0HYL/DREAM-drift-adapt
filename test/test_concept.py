#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_concept.py
@Time    :   2023/08/18 21:52:53
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import unittest
from . import generate_dummy_behavioral_data, set_gpu_id
set_gpu_id(0)
import tensorflow as tf
from dream.concept_learning import train_dream_model


class TestConcept(unittest.TestCase):
    def test_concept_learn(self):
        n_classes = 8
        input_dim = 1000
        data, behavior_labels, family_labels = generate_dummy_behavioral_data(input_dim=input_dim, n_classes=n_classes)
        # family_labels = tf.keras.utils.to_categorical(family_labels, num_classes=n_classes)
        # dataset = tf.data.Dataset.from_tensor_slices((data, behavior_labels, family_labels))

        # Dummy classifier
        classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),  # Intermediate layer (z)
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        n_c = behavior_labels.shape[-1]
        # Dummy encoder
        encoder_input = tf.keras.layers.Input(shape=(128,))
        encoded = tf.keras.layers.Dense(n_c, activation='relu')(encoder_input)
        encoder = tf.keras.Model(encoder_input, encoded)

        # Dummy decoder
        decoder_input = tf.keras.layers.Input(shape=(n_c,))
        decoded = tf.keras.layers.Dense(128, activation='relu')(decoder_input)
        decoder = tf.keras.Model(decoder_input, decoded)

        # Train the models
        # train_dream_model(data, family_labels, behavior_labels, classifier, 1, encoder, decoder, epochs=1)
        train_dream_model(data, family_labels, behavior_labels, classifier, encoder, decoder, split_layer_idx=1, epochs=1)
        