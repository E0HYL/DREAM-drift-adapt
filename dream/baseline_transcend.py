#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   baseline_transcend.py
@Time    :   2024/01/08 20:05:27
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2024
@Desc    :   adapted for DNN with arbitary number of output classes
'''

# here put the import lib
import os
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold

from dream.active_learning import retrain_model
from dream.classifiers import copy_compiled_model


def get_transcend_scores(X_train, y_train, X_test, model, train_epoch, model_path, num_folders=10):
    logging.debug(f'Running Transcendent CCE with {num_folders} folders')
    skf = StratifiedKFold(n_splits=num_folders, shuffle=True, random_state=21) 
    folds_generator = ({ 
        'X_train': X_train[train_index],
        'y_train': y_train[train_index],
        'X_cal': X_train[cal_index],
        'y_cal': y_train[cal_index],
        'X_test': X_test,
        'model': copy_compiled_model(model),
        'epoch': train_epoch,
        'model_path': model_path.replace('.h5', f'_fold-{idx}.h5'),
        'idx': idx
    } for idx, (train_index, cal_index) in enumerate(skf.split(X_train, y_train)))
    
    pval_lst = []
    for fold in folds_generator:   
        res = parallel_ice(fold)
        pval_lst.append(res)
    
    if X_test is not None:
        pval_arr = np.array(pval_lst)
        result_arr = np.median(pval_arr, axis=0)
        return 1 - result_arr # the higher, the more drifting


def parallel_ice(folds_gen): 
    X_train, y_train = folds_gen['X_train'], folds_gen['y_train']
    X_cal, y_cal = folds_gen['X_cal'], folds_gen['y_cal']
    model, epoch = folds_gen['model'], folds_gen['epoch'] 
    model_path = folds_gen['model_path']
    idx = folds_gen['idx'] 
    
    logging.debug('[Fold {}] Starting calibration...'.format(idx))  
    cal_results_dict = train_calibration_ice( 
            model=model,
            train_epochs=epoch,
            X_proper_train = X_train,
            X_cal = X_cal,
            y_proper_train = y_train,
            y_cal = y_cal,
            fold_index = 'cce_{}'.format(idx),
            model_path = model_path
    ) 

    X_test = folds_gen['X_test']
    if X_test is None:
        return None
    logging.debug('Calculating p-vals for test data')
    ncms_cal = cal_results_dict['ncms_cal']
    model = cal_results_dict['model']
    
    pred_test = model.predict(X_test).argmax(-1)
    ncms_test = get_class_uncertainty(model, X_test, pred_test)
    p_val_test = compute_p_values_cred(
        train_ncms=ncms_cal,
        groundtruth_train=y_cal,
        test_ncms=ncms_test,
        y_test=pred_test)
    
    result_arr = np.array(p_val_test)
    
    return result_arr


def get_class_uncertainty(model, data_batch, label_batch):
    """
    Calculate the uncertainty (negative probability) of a model's predictions for a batch of data points
    with respect to a batch of class labels.

    Parameters:
    - model: The neural network model that produces softmax output.
    - data_batch: A batch of input data points for which you want to calculate uncertainty.
    - label_batch: A batch of target class labels for which you want to calculate uncertainty.

    Returns:
    - uncertainties: A list of negative probabilities that the data in `data_batch` belongs to the specified classes in `label_batch`.
    """

    predictions_batch = model.predict(data_batch)
    uncertainties = - predictions_batch[np.arange(len(data_batch)), label_batch]

    return uncertainties


def train_calibration_ice(
        model, train_epochs, 
        X_proper_train, X_cal,
        y_proper_train, y_cal, fold_index, model_path):
    """Train calibration set (for a single fold).

    Args:
        X_proper_train (np.ndarray): Features for the 'proper training
            set' partition.
        X_cal (np.ndarray): Features for a single calibration set
            partition.
        y_proper_train (np.ndarray): Ground truths for the 'proper
            training set' partition.
        y_cal (np.ndarray): Ground truths for a single calibration set
            partition.
        fold_index: An index to identify the current fold (used for caching).

    Returns:
        dict: Fold results

    """
    # Train model with proper training
    if train_epochs > 0:
        retrain_model(model, (X_proper_train, y_proper_train), train_epochs, model_path, False, verbose=0, alternate_monitor='loss')
    else: assert os.path.exists(model_path)
    model.load_weights(model_path)

    # Compute ncm scores for calibration fold
    logging.debug('Computing cal ncm scores for fold {}...'.format(fold_index))
    ncms_cal_fold = get_class_uncertainty(model, X_cal, y_cal)

    return {
        'ncms_cal': ncms_cal_fold,  # Calibration NCMs
        'model': model
    }


def compute_p_values_cred(train_ncms, groundtruth_train, test_ncms, y_test):
    """
    Compute credibility p-values for an entire array of test NCMs.

    Args:
        train_ncms (np.ndarray): An array of training NCMs to compare the
            reference points against.
        groundtruth_train (np.ndarray): An array of ground truths corresponding
            to `train_ncms`.
        test_ncms (np.ndarray): An array of test NCMs for which to compute the
            p-values.
        y_test (np.ndarray): An array of ground truth or predicted labels
            corresponding to `test_ncms`.

    Returns:
        list: A list of p-values for each test NCM w.r.t. `train_ncms`.
    """
    # Create masks for points with the same ground truth as each test point
    same_class_masks = [groundtruth_train == y for y in y_test]

    cred = []
    for ncm, same_class_mask in zip(test_ncms, same_class_masks):
        # Count the number of points with greater NCMs than the test NCM
        greater_ncms = np.sum((train_ncms >= ncm) & same_class_mask)

        # Calculate the p-value for the test NCM
        total_same_class_points = np.sum(same_class_mask)
        single_cred_p_value = greater_ncms / total_same_class_points
        cred.append(single_cred_p_value)

    return cred 
