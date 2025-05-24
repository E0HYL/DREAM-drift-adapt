#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   active_learning.py
@Time    :   2023/08/18 09:28:52
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
from dream.train_utils import form_tf_data, create_checkpoint
from dream.evaluate import evaluate_model, multi2binary_malware_label


def evaluate_binary_detector(y_pred, y_true, anomaly_scores, top_k=None):
    predicted_labels = np.argmax(y_pred, axis=1)
    # Convert y_true to binary (0 or 1)
    y_true = (y_true > 0).astype(int)
    wrong_predictions = (predicted_labels != y_true).astype(int)
    if top_k is not None:
        # Sort the anomaly scores while keeping track of original indices
        sorted_indices = np.argsort(anomaly_scores)[::-1]
        top_k_indices = sorted_indices[:top_k]
        num_wrong_in_top_k = wrong_predictions[top_k_indices].sum()
        return num_wrong_in_top_k
    else:
        logging.info(f"{wrong_predictions.sum()}/{len(predicted_labels)} drift samples are correctly sampled, Goodware {wrong_predictions[y_true==0].sum()}/{len(y_true)-y_true.sum()}, Malware {wrong_predictions[y_true==1].sum()}/{y_true.sum()}.")


# Example uncertainty sampler
def sample_uncertain_instances(model, X_test, num_or_percentage, instance_id=2, **kwargs):
    # For multi-class classification, we can use entropy as a measure of uncertainty
    probs = model.predict(X_test)
    if instance_id == 0:
        entropies = probs[:, -1] # 
    elif instance_id == 1:
        entropies = 1 - (probs).max(-1) 
    elif instance_id == 2:
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    if 0 < num_or_percentage <= 1:  # It's a percentage
        num_samples = int(num_or_percentage * len(X_test))
    else:
        num_samples = int(min(num_or_percentage, len(X_test)))
    uncertain_indices = np.argsort(entropies)[-num_samples:]
    y_true = kwargs.get('y_test', None)
    if y_true is not None:
        evaluate_binary_detector(probs[uncertain_indices], y_true[uncertain_indices], entropies[uncertain_indices])
    return uncertain_indices, None


def sample_dream_uncertainty(model, X_test, num_or_percentage, **kwargs):
    entropies = model.get_drift_scores(X_test)
    num_samples = int(min(num_or_percentage, len(X_test)))
    uncertain_indices = np.argsort(entropies)[-num_samples:]
    y_true = kwargs.get('y_test', None)
    if y_true is not None:
        probs, y_concept_pred = model.get_predictions(X_test)
        evaluate_binary_detector(probs[uncertain_indices], y_true[uncertain_indices], entropies[uncertain_indices])
    return uncertain_indices, None


def hcc_sampler(model, X_test, num_or_percentage, **kwargs):
    X_cal = kwargs.get('X_train')
    y_cal = kwargs.get('y_train')
    classifier = kwargs.get('classifier')
    logging.info('Using hcc sampler with training data')
    n_neighbors = kwargs.get('num_near_neighbors', None)
    f_neighbors = kwargs.get('num_far_neighbors', None)
    # ContrastiveDetector
    anomaly_scores = model.detect_drift(X_cal, y_cal, X_test, classifier=classifier, num_near_neighbors=n_neighbors, num_far_neighbors=f_neighbors)
    if isinstance(anomaly_scores, tuple):
        anomaly_scores, selected_neighbors = anomaly_scores
    else:
        selected_neighbors = None
    anomaly_scores = np.array(anomaly_scores)[:, 0]
    if 0 < num_or_percentage <= 1:  # It's a percentage
        num_samples = int(num_or_percentage * len(X_test))
    else:
        num_samples = int(min(num_or_percentage, len(X_test)))
    uncertain_indices = np.argsort(anomaly_scores)[-num_samples:]
    if selected_neighbors is not None:
        selected_neighbors = np.array(selected_neighbors)
        # selected_training = np.concatenate(selected_neighbors[uncertain_indices], axis=0)
        selected_training = np.unique(selected_training)
        selected_training = np.random.choice(selected_training, num_samples*2, replace=False)
    else:
        selected_training = None
    y_true = kwargs.get('y_test', None)
    if y_true is not None:
        probs = model.predict(X_test)
        evaluate_binary_detector(probs[uncertain_indices], y_true[uncertain_indices], anomaly_scores[uncertain_indices])
    return uncertain_indices, selected_training


def retrain_model(model, train_data, retrain_epoch, model_path=None, incremental=True, **kwargs):
    train_data = form_tf_data(train_data, batch_size=kwargs.pop('batch_size', 32), buffer_size=100)
    monitor = kwargs.pop('monitor', 'sparse_categorical_accuracy')
    mode = kwargs.pop('mode', 'max')
    cp_callback = create_checkpoint(model_path, monitor=monitor, mode=mode, verbose=kwargs.get('verbose', 0), alternate_monitor=kwargs.pop('alternate_monitor', None))
    logging.info(f"Retraining with {'incremental' if incremental else 'masked'} data for {retrain_epoch} epochs...")
    model.fit(train_data, epochs=retrain_epoch, callbacks=cp_callback, **kwargs)


def sample_evenly(X, y, y_concept=None, num_sample=None, check_binary=False):   
    if check_binary:
        y = multi2binary_malware_label(y)
    # Identify unique classes
    unique_classes = np.unique(y)
    # Find the minimum class count if num_sample is not provided
    if num_sample is None:
        num_sample = min(np.bincount(y))
    
    sampled_indices = []
    # Sample evenly for each class
    for cl in unique_classes:
        class_indices = np.where(y == cl)[0] 
        cl_num_sample = num_sample*2 if (check_binary and cl==0) else num_sample
        num_sample = min(len(class_indices), cl_num_sample)
        sampled_class_indices = np.random.choice(class_indices, num_sample, replace=False)
        sampled_indices.extend(sampled_class_indices)
    
    np.random.shuffle(sampled_indices)
    X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]
    if y_concept is not None:
        return X_sampled, y_sampled, y_concept[sampled_indices]
    return X_sampled, y_sampled, None


'''
Active learning with fixed performance thresholds
    Select the most 1% uncertain samples to retrain the classifier, and then gradually increase the percentage by 1% until the F1 score reaches the threshold Th. 
    The initial model is trained on X_train, new samples are selected from X_test. **kwargs are passed to model.fit() in retraining, can be used to specify the batch_size/verbose/...
    Return the number of retrain times and the total number of new samples added.
'''
def active_learning_with_threshold(initial_model, X_train, y_train, X_test, y_test, y_concept=None, Th=0.9, percent_step=0.01, retrain_epoch=50, uncertainty_sampler=sample_uncertain_instances, old_model=None, del_sample=True, model_path=None, **kwargs):
    # Initialize 
    retrain_times = 0
    total_samples_added = 0
    total_new_samples = len(X_test)
    percentage = percent_step
    # Calculate initial F1 score
    ini_f1 = evaluate_model(model, X_test, y_test, ['f1'])
    f1 = ini_f1
    # If False, evaluation data are consistent (default to True: primarily to maintain the principle that models should be evaluated on unseen data to get a realistic sense of their performance.)
    if not del_sample: 
        ini_X_test = X_test
        ini_y_test = y_test
    if y_concept is not None:
        y_train_concept, y_test_concept = y_concept
    
    # Get the indices of the top uncertain samples using the sampler function
    old_model = initial_model if old_model is None else old_model
    uncertain_indices = uncertainty_sampler(old_model, X_test, percent_step)
    model = initial_model
    while f1 < Th:
        # Sample retraining data from previous training data
        sampled_X_train, sampled_y_train, sampled_y_concept_train = sample_evenly(X_train, y_train, y_concept=y_concept[0] if y_concept is not None else None, num_sample=len(uncertain_indices))
        _X_train = np.vstack([sampled_X_train, X_test[uncertain_indices]])
        _y_train = np.hstack([sampled_y_train, y_test[uncertain_indices]])
        if y_concept is None:
            retrain_data = (_X_train, _y_train)
        else:
            _y_train_concept = np.vstack([sampled_y_concept_train, y_test_concept[uncertain_indices]]) 
            retrain_data = (_X_train, _y_train, _y_train_concept)
            # Add and Remove for y_*_concept
            y_train_concept = np.vstack([y_train_concept, y_test_concept[uncertain_indices]])
            y_test_concept = np.delete(y_test_concept, uncertain_indices, axis=0)
        # Add selected samples to traing data (X_train and y_train)
        X_train = np.vstack([X_train, X_test[uncertain_indices]])
        y_train = np.hstack([y_train, y_test[uncertain_indices]]) 
        # Remove the selected samples from testing data (X_test and y_test)
        X_test = np.delete(X_test, uncertain_indices, axis=0)
        y_test = np.delete(y_test, uncertain_indices, axis=0)
        # Update the total samples added
        total_samples_added += len(uncertain_indices)
        
        # Retrain the model
        retrain_model(model, retrain_data, retrain_epoch, model_path, **kwargs)
        retrain_times += 1
        # Calculate F1 score again
        if model_path is not None: model.load_weights(model_path)
        f1 = evaluate_model(model, ini_X_test, ini_y_test, ['f1']) if not del_sample \
            else evaluate_model(model, X_test, y_test, ['f1'])  
        
        # Increase the percentage for the next iteration
        percentage += percent_step
        print(f"\r[{'='*int(percentage / percent_step)}] Retrain {retrain_times} times, F1 reached {'%.3f'%f1}", end='', flush=True)
        if percentage > 1:
            AssertionError(f'Fail to reach the F1 of {Th}')
        uncertain_indices = uncertainty_sampler(model, X_test, percent_step)
    
    logging.info(f"\nF1 {'%.3f'%ini_f1}->{'%.3f'%f1} (>{Th}), Samples used for updating is {total_samples_added}/{total_new_samples}.")
    return retrain_times, total_samples_added


'''
Active learning with fixed sample budgets
    Fix the number of newly selected samples from X_test
    Return the new model and its performance (F1, accuracy) on the remaining test set.
'''
def active_learning_with_budget(initial_model, X_train, y_train, X_test, y_test, y_concept=None, budget=50, retrain_epoch=50, model_verbose=False, uncertainty_sampler=sample_uncertain_instances, old_model=None, del_sample=True, model_path=None, num_family=8, sample=True, lazy_run=False, **kwargs): 
    check_binary = kwargs.pop('check_binary', False)
    ini_f1_train, ini_acc_train = evaluate_model(initial_model, X_train, y_train, y_concept=y_concept[0] if y_concept is not None else None, check_binary=check_binary)
    ini_f1, ini_acc = evaluate_model(initial_model, X_test, y_test, y_concept=y_concept[1] if y_concept is not None else None, check_binary=check_binary)
    if check_binary:
        ini_err_rate = evaluate_model(initial_model, X_test, y_test, y_concept=y_concept[1] if y_concept is not None else None, check_binary=check_binary, metric_name=['fpr', 'fnr'], legacy_format=False)
    # Get the indices of the samples to be added using the sampler function
    old_model = initial_model if old_model is None else old_model
    uncertain_indices, training_indices = uncertainty_sampler(old_model, X_test, budget, X_train=X_train, y_train=y_train, classifier=initial_model, y_test=y_test if check_binary else None)
    if training_indices is not None:
        sampled_X_train, sampled_y_train = X_train[training_indices], y_train[training_indices]
        sampled_y_concept_train = y_concept[0][training_indices] if y_concept is not None else None
    elif not sample: 
        sampled_X_train, sampled_y_train = X_train, y_train
        sampled_y_concept_train = y_concept[0] if y_concept is not None else None
    else:
        sampled_X_train, sampled_y_train, sampled_y_concept_train = sample_evenly(X_train, y_train, y_concept=y_concept[0] if y_concept is not None else None, num_sample=len(uncertain_indices))
    if num_family is not None:
        logging.info(f"{(y_test[uncertain_indices] == num_family - 1 ).sum()}/{budget} drift samples are correctly sampled.")
    
    # Add these samples to X_train and y_train
    X_train = np.vstack([sampled_X_train, X_test[uncertain_indices]])
    y_train = np.hstack([sampled_y_train, y_test[uncertain_indices]])
    if del_sample:
        # Remove the selected samples from X_test and y_test
        X_test = np.delete(X_test, uncertain_indices, axis=0)
        y_test = np.delete(y_test, uncertain_indices, axis=0)
    
    model = initial_model
    # Retrain the model
    if y_concept is None:
        train_data = (X_train, y_train)
    else:
        y_train_concept = np.vstack([sampled_y_concept_train, y_concept[1][uncertain_indices]])
        y_test_concept = np.delete(y_concept[1], uncertain_indices, axis=0)
        train_data = (X_train, y_train, y_train_concept)
    retrain_flag = False
    if not lazy_run or not os.path.exists(model_path):
        retrain_model(model, train_data, retrain_epoch, model_path=model_path, **kwargs)
        retrain_flag = True

    # Evaluate the model on the (remaining) X_test
    if model_path is not None: 
        logging.info(f'Finish updating: {model_path}')
        model.load_weights(model_path)
    f1, accuracy = evaluate_model(model, X_test, y_test, y_concept=y_test_concept if y_concept is not None else None, check_binary=check_binary)
    if retrain_flag:
        f1_train, accuracy_train = evaluate_model(model, X_train, y_train, y_concept=y_train_concept if y_concept is not None else None, check_binary=check_binary)
        log_msg = f"""Results for Sample budget [{budget}]:
            Train | F1: {'%.3f'%ini_f1_train} -> {'%.3f'%f1_train} | Acc: {'%.3f'%ini_acc_train[0]} -> {'%.3f'%accuracy_train[0]}
            Test  | F1: {'%.3f'%ini_f1} -> {'%.3f'%f1} | Acc: {'%.3f'%ini_acc[0]} -> {'%.3f'%accuracy[0]}"""
        if check_binary:
            err_rate = evaluate_model(model, X_test, y_test, y_concept=y_test_concept if y_concept is not None else None, check_binary=check_binary, metric_name=['fpr', 'fnr'], legacy_format=False)
            log_msg += f"""
            Test error | FPR: {'%.3f'%ini_err_rate['fpr']} -> {'%.3f'%err_rate['fpr']} | FNR: {'%.3f'%ini_err_rate['fnr']} -> {'%.3f'%err_rate['fnr']}"""
        if y_concept is not None:
            log_msg += f"""
            --------------------------------
            ConceptAcc: Train {'%.3f'%ini_acc_train[1]} -> {'%.3f'%accuracy_train[1]} | Test {'%.3f'%ini_acc[1]} -> {'%.3f'%accuracy[1]}"""
    else:
        log_msg = f"""Results for Sample budget [{budget}]:
            Test  | F1: {'%.3f'%ini_f1} -> {'%.3f'%f1} | Acc: {'%.3f'%ini_acc[0]} -> {'%.3f'%accuracy[0]}"""
    logging.info(log_msg)
    
    return (model, uncertain_indices, X_test) if model_verbose else (f1, accuracy[0])


"""
Continuous active learning with time budget
    Divide X_test by corresponding date_time into n slots. For i-th iteration, update the model with samples slected from the i-th slot with the uncertainty_sampler.
    Return the model performance (F1, accuracy) at each iteration.
"""
def continuous_active_learning(initial_model, X_train, y_train, X_test, y_test, test_date_time, budget=200, num_slots=5, retrain_epoch=50, uncertainty_sampler=sample_uncertain_instances, **kwargs):
    # Evaluate the initial model's performance on X_train
    f1_initial, accuracy_initial = evaluate_model(initial_model, X_train, y_train)
    # Start the performance list with the initial model's performance
    f1_slots = [f1_initial]
    acc_slots = [accuracy_initial]
    
    # Sort X_test and y_test by date_time
    sorted_indices = np.argsort(test_date_time)
    X_test_sorted = X_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    
    # Divide the sorted X_test into n slots
    slot_size = len(X_test_sorted) // num_slots

    for i in range(num_slots):
        # Determine the start and end indices for the current slot
        start_idx = i * slot_size
        end_idx = start_idx + slot_size if i != num_slots - 1 else len(X_test_sorted)
        X_slot = X_test_sorted[start_idx:end_idx]
        y_slot = y_test_sorted[start_idx:end_idx]

        # Get the indices of the samples to be added using the sampler function
        uncertain_indices = uncertainty_sampler(initial_model, X_slot, budget)
        # Add these samples to X_train and y_train
        X_train = np.vstack([X_train, X_slot[uncertain_indices]])
        y_train = np.hstack([y_train, y_slot[uncertain_indices]])
        # Remove the selected samples from X_slot and y_slot
        X_slot = np.delete(X_slot, uncertain_indices, axis=0)
        y_slot = np.delete(y_slot, uncertain_indices, axis=0)
        
        # Retrain the model
        initial_model.fit(X_train, y_train, epochs=retrain_epoch, **kwargs)
        # Evaluate the model on the current slot
        f1, accuracy = evaluate_model(initial_model, X_slot, y_slot)
        f1_slots.append(f1)
        acc_slots.append(accuracy)

    return f1_slots, acc_slots
