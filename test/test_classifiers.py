#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_classifiers.py
@Time    :   2023/07/26 15:30:26
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import unittest
from . import generate_dummy_data, set_gpu_id
set_gpu_id(0)
import tensorflow as tf
from dream.classifiers import DAMDModel, MarkovCNN, DrebinMLP
from dream.train_utils import train_model

class TestCls(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.num_classes = 2

    def get_data_classifier(self, data_name='damd'):
        if data_name == 'damd':
            vocab_size = 258
            x_train, y_train = generate_dummy_data(type='text', num_classes=self.num_classes, num_samples=100, vocab_size=vocab_size)
            classifier = DAMDModel(no_tokens=vocab_size, no_labels=self.num_classes)
        elif data_name == 'mamadroid':
            x_train, y_train = generate_dummy_data(type='image', num_classes=self.num_classes, num_samples=100)
            input_shape = x_train.shape[1:]
            classifier = MarkovCNN(input_shape=input_shape, num_classes=self.num_classes)
        elif data_name == 'drebin':
            x_train, y_train = generate_dummy_data(type='tabular', num_classes=self.num_classes, num_samples=100)
            classifier = DrebinMLP()
        classifier.build((None,) + x_train.shape[1:])
        classifier.summary()
        return x_train, y_train, classifier
    
    def test_classifier(self):
        for data_name in ['damd', 'mamadroid', 'drebin']:
            x_train, y_train, classifier = self.get_data_classifier(data_name=data_name)
            y_train = tf.one_hot(y_train, self.num_classes)
            loss = 'binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy'
            train_model(x_train, y_train, classifier, loss=loss)
            classifier.predict(x_train)
