#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_cade.py
@Time    :   2023/07/24 13:34:05
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import unittest
from dream.autoencoder import TabularAutoencoder, TextAutoencoder, ImageAutoencoder, SeqAutoencoder, Conv1DTextAutoencoder
from dream.baseline_cade import epoch_train
from . import generate_dummy_data


class TestCADE(unittest.TestCase):
    def get_data_ae(self, type):
        print(f'{type}: construct autoencoder')
        # Assume we have 3 classes, 1000 samples
        x_train, y_train = generate_dummy_data(type, num_classes=3, num_samples=1000)
        input_shape = x_train.shape[1:]
        if type == 'tabular':
            autoencoder = TabularAutoencoder(input_dim=input_shape[0], encoding_dim=3)
        elif type == 'text':
            # autoencoder = TextAutoencoder(vocab_size=10000, embedding_dim=50, sequence_length=100, encoding_dim=30)  
            autoencoder = Conv1DTextAutoencoder(vocab_size=10000, embedding_dim=50, sequence_length=100) 
        elif type == 'image':
            autoencoder = ImageAutoencoder(input_shape=input_shape, encoding_dim=64)
        elif type == 'sequence':
            autoencoder = SeqAutoencoder(input_shape=input_shape)
        else:
            raise ValueError(f"Unknown type {type}")
        return x_train, y_train, autoencoder, input_shape

    def test_cade_train(self):
        for ae_type in ['text']: # 'image', 'text', 'tabular', 'sequence'
            x_train, y_train, autoencoder, input_shape = self.get_data_ae(ae_type)
            autoencoder.build((None,)+input_shape)
            print(autoencoder.summary())
            epoch_train(x_train=x_train, y_train=y_train, epochs=2, batch_size=16,input_shape=input_shape, autoencoder=autoencoder)
