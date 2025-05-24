#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   load_data.py
@Time    :   2023/07/26 19:37:29
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
import h5py
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dream.autoencoder import TabularAutoencoder, Conv1DTextAutoencoder, ImageAutoencoder
from dream.classifiers import DAMDModel, MarkovCNN, DrebinMLP, MarkovMLP


import logging
logger = logging.getLogger(__name__)


def get_dataset_family(dataset_name, count=False):
    metafile = f'data/{dataset_name}_metadata.csv'
    meta = pd.read_csv(metafile)
    family_count = meta.family.value_counts()
    if dataset_name == 'binarycls':
        family_names = ['Benign'] + pd.read_csv(f'data/{dataset_name}/{dataset_name}_family_behavior.csv')['Family']
        family_count['Benign'] = np.nan # TODO:count benign samples
        family_count = family_count.reindex(family_names)
    family_names = list(family_count.items()) if count else family_count.keys().tolist()  
    return family_names         


def pad_damd_seq(data, maxlen=200000, padding='post', truncating='post'):
    return pad_sequences(data, maxlen=maxlen, padding=padding, truncating=truncating)


def load_data_model(data_name='drebin', feature_name='damd', model_type='basic', behavior_flag=False, num_cls_class=None, **kwargs):
    analyzer = feature_name.capitalize()
    if data_name == 'binarycls':
        _num_family = 2
        family_flag = False if model_type == 'basic' else True
        year = kwargs.pop('split_year', 2015)
        x_train, y_train, x_test, y_test = load_data_with_year_split(year, analyzer=analyzer, behavior_label=behavior_flag, family_label=family_flag, end_year=2020)
    else:
        newfamily = kwargs.pop('newfamily', 0)
        num_family = kwargs.pop('num_family', 8)
        _num_family = num_family - 1 if num_cls_class is None else num_cls_class # basic: number of families in training set
        x_train, y_train, x_test, y_test = load_data_with_hold_out(data_name=data_name, newfamily=newfamily, num_family=num_family, analyzer=analyzer, behavior_label=behavior_flag)
    
    assert model_type in ['basic', 'cade', 'dream']
    if feature_name == 'damd':
        vocab_size = 218 # max(x_train.max(), x_test.max()) + 1
        classifier = DAMDModel(no_tokens=vocab_size, no_labels=_num_family)
        autoencoder = Conv1DTextAutoencoder(vocab_size=vocab_size, sequence_length=x_train.shape[-1], emb_layer=kwargs.pop('emb_layer', True), filters=kwargs.pop('filters', 64))
    elif feature_name == 'mamadroid':
        mlp_arch = kwargs.pop('markov_mlp', True)
        input_shape = x_train.shape[1:]
        if mlp_arch:
            classifier = MarkovMLP(dims=[1000, 200, _num_family])
            autoencoder = TabularAutoencoder(input_dim=np.multiply(*input_shape), hidden_dim=2048, encoding_dim=128)
            x_train = (x_train*10).reshape(len(x_train), -1)
            if type(x_test) == dict:
                for k in x_test:
                    v = x_test[k]
                    x_test[k] = (v*10).reshape(len(v), -1)
            else:
                x_test = (x_test*10).reshape(len(x_test), -1)
        else:
            classifier = MarkovCNN(input_shape=input_shape, num_classes=_num_family)
            autoencoder = ImageAutoencoder(input_shape=input_shape, encoding_dim=128)
            x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)
    elif feature_name == 'drebin':
        num_features = x_train.shape[-1]
        classifier = DrebinMLP(dims=[num_features, 100, 30, _num_family])
        autoencoder = TabularAutoencoder(input_dim=num_features, hidden_dim=512, encoding_dim=32)
    else:
        raise ValueError(f'no classifier named {feature_name}')
    if model_type == 'basic':
        model = classifier
    elif model_type == 'cade':
        model = autoencoder
    elif model_type == 'dream':
        model = (classifier, autoencoder)
    
    return x_train, y_train, x_test, y_test, model


def load_data_with_hold_out(data_name='drebin', newfamily=0, num_family=8, folder='data', analyzer='Damd', behavior_label=True):
    logger.debug('Loading ' + f'{data_name}/{analyzer}/{newfamily}' + ' feature vectors and labels...')
    filepath = os.path.join(folder, data_name, analyzer, f'{newfamily}.npz')
    obj_flag = (analyzer == 'Damd') # sequences with unfixed length
    data = np.load(filepath, allow_pickle=obj_flag)
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    if obj_flag:
        X_train = pad_damd_seq(X_train)
        X_test = pad_damd_seq(X_test)
    if behavior_label:
        family_behavior = pd.read_csv(os.path.join(folder, data_name, f'{data_name}_family_behavior.csv'))
        yb_train = np.array([family_behavior.iloc[y, 1:] for y in y_train], dtype=int)
        yb_test = np.array([family_behavior.iloc[y, 1:] for y in y_test], dtype=int)
        non_empty_flag = (yb_train.sum(0) + yb_test.sum(0)) > 0
        yb_train = yb_train[:, non_empty_flag]
        yb_test = yb_test[:, non_empty_flag]

    logger.debug(f'before label adjusting: y_train: {Counter(y_train)}\n  y_test: {Counter(y_test)}')

    '''transform training set to continuous labels, always use the biggest label as the unseen family'''
    le = LabelEncoder()
    y_train_prime = le.fit_transform(y_train)
    mapping = {}
    for i in range(len(y_train)):
        mapping[y_train[i]] = y_train_prime[i]  # mapping: real label -> converted label

    logger.debug(f'LabelEncoder mapping: {mapping}')

    y_test_prime = np.zeros(shape=y_test.shape, dtype=np.int32)
    for i in range(len(y_test)):
        if y_test[i] not in y_train:  # new family
            y_test_prime[i] = num_family - 1
        else:
            y_test_prime[i] = mapping[y_test[i]]

    y_train_prime = np.array(y_train_prime, dtype=np.int32)
    logger.debug(f'after relabeling training: {Counter(y_train_prime)}')
    logger.debug(f'after relabeling testing: {Counter(y_test_prime)}')

    if behavior_label:
        logger.debug(f'Loaded train data: {X_train.shape}, ({y_train_prime.shape}, {yb_train.shape})')
        logger.info(f'`Train`: Family {Counter(y_train_prime)}, {yb_train.shape[-1]}-Behavior Counter {np.where(yb_train != -1, yb_train, 0).sum(0)}')
        logger.debug(f'Loaded test data: {X_test.shape}, ({y_test_prime.shape}, {yb_test.shape})')
        logger.info(f'`Test`: Family {Counter(y_test_prime)}, {yb_train.shape[-1]}-Behavior Counter {np.where(yb_test != -1, yb_test, 0).sum(0)}')
        return X_train, (y_train_prime, yb_train), X_test, (y_test_prime, yb_test)
    else:
        logger.debug(f'Loaded train: {X_train.shape}, {y_train.shape}')
        logger.debug(f'Loaded test: {X_test.shape}, {y_test.shape}')
        logger.debug(f'y_train: {Counter(y_train)}')
        logger.debug(f'y_test: {Counter(y_test)}')
        return X_train, y_train_prime, X_test, y_test_prime


def load_data_with_year_range(year_range, analyzer, folder='data', behavior=True, family=True):
    X = []
    y = []
    yb = []
    obj_flag = (analyzer == 'Damd')
    for year in year_range:
        filepath = os.path.join(folder, 'binarycls', analyzer, f'%s_{year}.npy')
        malware = np.load(filepath%'malware', allow_pickle=obj_flag)
        benign = np.load(filepath%'benign', allow_pickle=obj_flag)
        if obj_flag:
            malware = pad_damd_seq(malware)
            benign = pad_damd_seq(benign)
        X.append(np.concatenate([malware, benign]))
        malware_labels = np.load(os.path.join(folder, 'binarycls/_labels', f'malware_{year}.npz'))
        if family:
            y.append(np.concatenate([malware_labels['family'], [0]*len(benign)]))
        else:
            y.append(np.concatenate([[1]*len(malware), [0]*len(benign)]))
        if behavior: 
            malware_yb = malware_labels['behavior']
            benign_yb = np.zeros((len(benign), )+malware_yb.shape[1:])
            yb.append(np.concatenate([malware_yb, benign_yb])) 
    return X, y, yb


def load_data_with_year_split(year=2015, start_year=2015, end_year=2020, folder='data', analyzer='Damd', family_label=True, behavior_label=True):
    train_years = range(start_year, year+1)
    test_years = range(year+1, end_year+1)
    train_data = load_data_with_year_range(train_years, analyzer=analyzer, folder=folder, behavior=behavior_label, family=family_label)
    test_data = load_data_with_year_range(test_years, analyzer=analyzer, folder=folder, behavior=behavior_label, family=family_label)
    if behavior_label:
        X_train, y_train, yb_train = [np.concatenate(data) for data in train_data]
        X_test, y_test, yb_test = [dict(zip(test_years, data)) for data in test_data]
        return X_train, (y_train, yb_train), X_test, (y_test, yb_test)
    else:
        X_train, y_train = [np.concatenate(data) for data in train_data[:-1]]
        X_test, y_test = [dict(zip(test_years, data)) for data in test_data[:-1]]
        return X_train, y_train, X_test, y_test


def get_model_path(data_name, feature_name, newfamily, model_type, config=False):
    model_folder = os.path.join('models', data_name, feature_name.capitalize(), model_type)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_path = os.path.join(model_folder, f'{newfamily}.h5')
    if config:
        config_path = os.path.join(model_folder, f'model_config.txt')
        return model_path, config_path
    return model_path


def load_trained_model(model_arch, data_name, feature_name, newfamily, input_shape, model_name='basic'):
    model_folder = os.path.join('models', data_name, feature_name.capitalize(), model_name)
    model_path = os.path.join(model_folder, f'{newfamily}.h5')
    assert os.path.exists(model_path), f'Model not found in {model_path}'
    model_arch.build((None,) + input_shape)
    try:
        logging.info(f'loading model weights from {model_path}')
        model_arch.load_weights(model_path)
    except Exception as e:
        with h5py.File(model_path, 'r') as file:
            weight_dict = check_weights_file(file)
        logging.info(f'Error when loading [{type(model_arch)}] weights, found weight dict:\n{weight_dict.keys()}')
        raise e
    return model_arch

    
def check_weights_file(file, weight_dict=None):
    if weight_dict is None:
        weight_dict = {}    
    for key in file.keys():
        item = file[key]
        if isinstance(item, h5py.Dataset):  # Found weights
            weight_dict[key] = item[:]
        elif isinstance(item, h5py.Group):  # Nested weights
            weight_dict[key] = {}
            check_weights_file(item, weight_dict[key])          
    return weight_dict   
 

def flatten_weights(weight_dict, parent_key='', ordered_weights=None):
    # to match model.weights
    if ordered_weights is None:
        ordered_weights = {}
    for k, v in weight_dict.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            flatten_weights(v, new_key, ordered_weights)
        else:
            ordered_weights[new_key] = v     
    return ordered_weights


def read_weights(fname):
    with h5py.File(fname, 'r') as file:
        wd = flatten_weights(check_weights_file(file))
    return wd


def remove_duplicates(vectors, padding_value=-1):
    max_length = max(len(vector) for vector in vectors)
    padded_data = [np.pad(np.array(vector), (0, max_length - len(vector)), 'constant', constant_values=padding_value) for vector in vectors]
    np_data = np.array(padded_data)
    unique_data, indices = np.unique(np_data, axis=0, return_index=True)
    return indices # np_data[np.sort(indices)]                                


def get_data(data_name, fea_name):
    data_path=f'./data/{data_name}/{fea_name}/7.npz'
    data=np.load(data_path, allow_pickle=(fea_name=='Damd'))
    X_train, X_test, y_train, y_test=data['X_train'],data['X_test'],data['y_train'],data['y_test']
    X=np.concatenate([X_train,X_test])
    y=np.concatenate([y_train, y_test])
    return X, y
