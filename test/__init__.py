#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2023/07/26 15:37:45
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
import numpy as np


def generate_dummy_data(type, num_classes=3, num_samples=1000, input_shape=None, **kwargs): 
    if type == 'tabular':
        # each sample with 10 features
        input_shape = (10,) if input_shape is None else input_shape
        x_train = np.random.normal(size=(num_samples,)+input_shape)
        y_train = np.random.randint(0, num_classes, size=(num_samples,)) 
    elif type == 'text':
        # indexed words, each a sequence of 100 words, and a vocabulary of 10000 words
        input_shape = (100,) if input_shape is None else input_shape
        vocab_size = kwargs.get('vocab_size', 10000)
        x_train = np.random.randint(0, vocab_size, size=(num_samples,)+input_shape).astype(int)
        y_train = np.random.randint(0, num_classes, size=(num_samples,))   
    elif type == 'image':
        # grayscale images, each 28x28 pixels
        input_shape = (28, 28, 1) if input_shape is None else input_shape
        x_train = np.random.normal(size=(num_samples,)+input_shape)
        y_train = np.random.randint(0, num_classes, size=(num_samples,)) 
    elif type == 'sequence':
        # one-hot encoded words
        input_shape = (100, 10) if input_shape is None else input_shape
        x_train = np.random.random(size=(num_samples,)+input_shape)
        y_train = np.random.randint(0, num_classes, size=(num_samples,))

    return x_train, y_train


def generate_dummy_behavioral_data(input_dim=784, n_c=5, n_classes=10, n_samples=3200):
    # Randomly generate data for inputs
    inputs = np.random.rand(n_samples, input_dim)
    # Randomly generate concept_binary_labels with some missing labels
    concept_binary_labels = np.random.choice([0, 1, -1], size=(n_samples, n_c))
    # Randomly generate classifier_labels and one-hot encode them
    classifier_labels = np.random.randint(0, n_classes, size=n_samples)
    return inputs, concept_binary_labels, classifier_labels


def set_gpu_id(id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)


def lst_to_str(lst):
    if len(lst) == 0:
        return ""
    elif len(lst) == 1:
        return f'[{str(lst[0])}]'
    elif len(lst) == 2:
        return f"[{lst[0]}] and [{lst[1]}]"
    else:
        return ", ".join(f'[{str(x)}]' for x in lst[:-1]) + f", and [{lst[-1]}]"
