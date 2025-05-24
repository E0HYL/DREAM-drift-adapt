#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_utils.py
@Time    :   2024/10/07 15:26:41
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2024
@Desc    :   None
'''

# here put the import lib
import numpy as np
from dream.load_data import load_data_with_hold_out, load_trained_model
from dream.concept_learning import DREAM
from dream.classifiers import DAMDModel, DrebinMLP, MarkovMLP, summarize_classifier
from dream.autoencoder import TabularAutoencoder, Conv1DTextAutoencoder
from dream.train_utils import ModelConfig 
from dream.baseline_cade import ContrastiveAutoencoder


def get_classifier(feature_name, num_family, input_shape=None, **kwargs):
    _num_family = num_family - 1
    if feature_name == 'damd':
        vocab_size = kwargs.get('vocab_size')
        classifier = DAMDModel(no_tokens=vocab_size, no_labels=_num_family)
    elif feature_name == 'mamadroid':
        classifier = MarkovMLP(dims=[1000, 200, _num_family])
    elif feature_name == 'drebin':
        num_features = kwargs.get('num_features')
        classifier = DrebinMLP(dims=[num_features, 100, 30, _num_family])
    else:
        raise ValueError(f'no feature named `{feature_name}`')
    if input_shape is not None:
        summarize_classifier(classifier, input_shape)
    return classifier


def get_dream_model(feature_name, num_family, built=True, data_name=None, **kwargs):
    classifier = get_classifier(feature_name, num_family, **kwargs)
    if feature_name == 'damd':
        vocab_size = kwargs.get('vocab_size')
        sequence_length = kwargs.get('sequence_length')
        autoencoder = Conv1DTextAutoencoder(vocab_size=vocab_size, sequence_length=sequence_length)
    elif feature_name == 'mamadroid':
        input_dim = kwargs.get('input_dim')
        autoencoder = TabularAutoencoder(input_dim=input_dim, hidden_dim=2048, encoding_dim=128)
    elif feature_name == 'drebin':
        num_features = kwargs.get('num_features')
        autoencoder = TabularAutoencoder(input_dim=num_features, hidden_dim=512, encoding_dim=32)
    else:
        raise ValueError(f'no feature named `{feature_name}`') 
    autoencoder.build()   
    if built:
        concept_cls = None if feature_name == 'mamadroid' else True
        return build_dream(data_name, feature_name, classifier, autoencoder, concept_cls)
    else:
        return classifier, autoencoder


def build_dream(data_name, feature_name, classifier, autoencoder, concept_cls=True):
    dream_config = ModelConfig().get_config(data_name, feature_name, 'dream')
    dream_model = DREAM(classifier, autoencoder.encoder, autoencoder.decoder, concept_cls=concept_cls, **dream_config)
    return dream_model


def load_data_with_trained_model(data_name, feature_name, newfamily, num_family, behavior_flag=True, dream=True):
    x_train, y_train, x_test, y_test = load_data_with_hold_out(data_name=data_name, newfamily=newfamily, num_family=num_family, analyzer=feature_name.capitalize(), behavior_label=behavior_flag)

    input_shape = x_train.shape[1:]
    if feature_name == 'damd':
        vocab_size = 218 # max(x_train.max(), x_test.max()) + 1
        seq_length = x_train.shape[-1]
        kwargs = {'input_shape': input_shape, 'sequence_length':seq_length, 'vocab_size':vocab_size}
    elif feature_name == 'mamadroid':
        input_dim = np.multiply(*input_shape)
        x_train = (x_train*10).reshape(len(x_train), -1)
        x_test = (x_test*10).reshape(len(x_test), -1)
        input_shape = x_train.shape[1:]
        kwargs = {'input_shape': input_shape, 'input_dim': input_dim}
    elif feature_name == 'drebin':
        num_features = x_train.shape[-1]
        kwargs = {'input_shape': input_shape, 'num_features':num_features}
    else:
        raise ValueError(f'no feature named `{feature_name}`')

    dream_model = get_dream_model(feature_name, num_family, built=dream, data_name=data_name, **kwargs)
    if dream:
        model = load_trained_model(dream_model, data_name, feature_name, newfamily, input_shape, 'dream')
    else:
        classifier, autoencoder = dream_model
        classifier = load_trained_model(classifier, data_name, feature_name, newfamily, input_shape, 'basic')
        cade = load_trained_model(ContrastiveAutoencoder(autoencoder), data_name, feature_name, newfamily, input_shape, 'cade')
        model = build_dream(data_name, feature_name, classifier, cade.autoencoder)

    return x_train, y_train, x_test, y_test, model
