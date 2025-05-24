#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   adapt_baselines.py
@Time    :   2024/08/27 11:31:11
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2024
@Desc    :   Inter-drift adaptation experiments with more samplers and training strategies
'''

# here put the import lib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser(description="Drift Adaptation Baselines")
parser.add_argument("--data_name", default="drebin", type=str, help="Data name")
parser.add_argument("--feature_name", default="drebin", type=str, help="Feature name (options: drebin, damd, mamadroid)")
parser.add_argument("--sampler_name", default="entropy", type=str, help="Sampler name (options: entropy, uncertainty, cade, transcendent)")
parser.add_argument("--budget", default=10, type=int, help="Number of selected drift samples for retraining")
parser.add_argument("--batch_size", default=32, type=int, help="Training batch size")
parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID")
parser.add_argument('--display', dest='display_mode', action='store_true', help='Display all results of current data, feature, and samplers')
args = parser.parse_args()

from test import set_gpu_id
set_gpu_id(args.gpu_id)
from dream import set_random_seed, set_gpu_growing
set_gpu_growing()
set_random_seed(0)

from dream.load_data import load_trained_model, get_dataset_family, get_model_path, load_data_model
from dream.active_learning import active_learning_with_budget, sample_evenly
from dream.classifiers import copy_compiled_model, summarize_classifier, modify_output
from dream.train_utils import clear_models, default_classifier_compile, form_tf_data, ModelConfig
from dream.concept_learning import ContraBase, DREAM
from dream.ood_detector import PseudoBatch
from dream.baseline_cade import detect_drift
from dream.baseline_transcend import get_transcend_scores
from dream.baseline_cade import ContrastiveAutoencoder


class DriftDetector():
    def __init__(self, detector_name, data_name=None, feature_name=None, model_tag=None) -> None:
        self.detector_name = detector_name
        if detector_name == 'transcendent':
            # model_tag = split_year if data_name == 'binarycls' else newfamily
            self.model_path = get_model_path(data_name, feature_name, model_tag, self.detector_name)

    def get_anomaly_scores(self, X_train, y_train, X_test):
        if self.detector_name == 'cade': # ContrastiveAutoencoder
            anomaly_scores, _ = detect_drift(X_train, y_train, X_test, self.model, batch_size=16)
        elif self.detector_name == 'transcendent': # Classifier
            anomaly_scores = get_transcend_scores(X_train, y_train, X_test, self.model, train_epoch=0, model_path=self.model_path)
        elif self.detector_name == 'uncertainty':
            probs = self.model.predict(X_test)
            anomaly_scores = 1 - (probs).max(-1)
        elif self.detector_name == 'entropy': # `vanilla`` in adapt_classifier script
            probs = self.model.predict(X_test)
            anomaly_scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        elif self.detector_name == 'hcc': # StackModel
            anomaly_scores = self.model.get_drift_scores(X_test, X_train, y_train, batch_size=16)
        elif self.detector_name == 'dream-':
            anomaly_scores = self.model.get_drift_scores(X_test)
        return anomaly_scores
    
    def drift_sampler(self, model, X_test, num_or_percentage, **kwargs):
        self.model = model
        X_cal = kwargs.get('X_train', None)
        y_cal = kwargs.get('y_train', None)
        anomaly_scores = self.get_anomaly_scores(X_cal, y_cal, X_test)
        num_samples = int(min(num_or_percentage, len(X_test)))
        uncertain_indices = np.argsort(anomaly_scores)[-num_samples:]
        return uncertain_indices, None


def adapt_classifier_arch(model, num_classes, **kwargs):
    compile_loss = kwargs.get('loss', 'sparse_categorical_crossentropy')
    compile_metrics = kwargs.get('metric', ['sparse_categorical_accuracy'])
    compile_lr = kwargs.get('lr', .0004)
    if kwargs.get('stack', False):
        model.classifier =  modify_output(model.classifier, num_classes, name=model.name)
    elif isinstance(model, DREAM):
        model.f1 = modify_output(model.f1, num_classes, name=model.f1.name)
    else:
        model = modify_output(model, num_classes, name=model.name)
        # print(model.layers[-1].weights[0].numpy().sum())
        # breakpoint() # To ensure the random state is consistent with `adapt_classifier` script
    # Freeze customized model before compilation
    model = default_classifier_compile(model, loss=compile_loss, metrics=compile_metrics, lr=compile_lr, debug=True)
    return model


def retrain_with_base_sampler(detector_name, data_name, feature_name, newfamily, num_family, lr, budget, retrain_epoch=50):
    _, _, _, _, classifier = load_data_model(
        data_name, feature_name, model_type='basic', newfamily=newfamily, num_family=num_family, behavior_flag=False)
    x_train, y_train, x_test, y_test, autoencoder = load_data_model(
        data_name, feature_name, model_type='cade', newfamily=newfamily, num_family=num_family, behavior_flag=False)
    
    sampler = DriftDetector(detector_name, data_name, feature_name, newfamily)
    uncertainty_sampler = sampler.drift_sampler
    input_shape = autoencoder._input_shape
    summarize_classifier(classifier, input_shape)
    classifier = load_trained_model(classifier, data_name, feature_name, newfamily, input_shape)
    
    if detector_name == 'cade':
        # load trained cade detector
        summarize_classifier(autoencoder, input_shape)
        detector_model = load_trained_model(ContrastiveAutoencoder(autoencoder), data_name, feature_name, newfamily, input_shape, 'cade').autoencoder
    elif detector_name == 'transcendent':
        # pass the classifier architecture
        detector_model = copy_compiled_model(classifier, copy_weights=False)
        del autoencoder
    elif detector_name == 'uncertainty' or detector_name == 'entropy':
        detector_model = classifier
        del autoencoder
    elif detector_name == 'dream-':
        summarize_classifier(autoencoder, input_shape)
        dream_config = ModelConfig().get_config(data_name, feature_name, 'dream')
        dream_model = DREAM(classifier, autoencoder.encoder, autoencoder.decoder, 
                            concept_cls=None if feature_name=="mamadroid" else True, **dream_config)
        detector_model = load_trained_model(dream_model, data_name, feature_name, newfamily, input_shape, 'dream')
    # print(np.random.get_state()[1].sum())
    model = adapt_classifier_arch(classifier, num_family, lr=lr) 

    result_folder = os.path.join('results', data_name, feature_name.capitalize(), f'_bud_{budget} (tuningFalse_retrain{retrain_epoch}_aeFalse)')
    model_path = os.path.join(result_folder, f'{newfamily}_updated_{detector_name}.h5')
    f1, accuracy = active_learning_with_budget(model, x_train, y_train, x_test, y_test, budget=budget, 
                                               verbose=0, retrain_epoch=retrain_epoch, 
                                               batch_size=args.batch_size, 
                                               old_model=detector_model, 
                                               model_path=model_path, num_family=num_family, 
                                               uncertainty_sampler=uncertainty_sampler, 
                                               lazy_run=False)

    consistent_rng_handler(x_train, y_train, x_test, y_test)
    del classifier, model, uncertainty_sampler, detector_model
    clear_models()
    gc.collect()
    
    return [newfamily, detector_name, f1, accuracy]


def retrain_concept_with_base_sampler(detector_name, data_name, feature_name, newfamily, num_family, lr, budget, retrain_epoch=50):
    x_train, y_train, x_test, y_test, (classifier, autoencoder) = load_data_model(
        data_name, feature_name, model_type='dream', newfamily=newfamily, num_family=num_family, behavior_flag=True)
    y_concept = (y_train[1], y_test[1])
    y_train = y_train[0]
    y_test = y_test[0]
 
    sampler_name = detector_name[:-1]
    sampler = DriftDetector(sampler_name, data_name, feature_name, newfamily)
    uncertainty_sampler = sampler.drift_sampler
    input_shape = autoencoder._input_shape
    summarize_classifier(classifier, input_shape)
    summarize_classifier(autoencoder, input_shape)
    dream_config = ModelConfig().get_config(data_name, feature_name, 'dream')
    dream_model = DREAM(classifier, autoencoder.encoder, autoencoder.decoder, 
                        concept_cls=None if feature_name=="mamadroid" else True, **dream_config)
    dream_model = load_trained_model(dream_model, data_name, feature_name, newfamily, input_shape, 'dream')

    if sampler_name == 'cade':
        detector_model = load_trained_model(ContrastiveAutoencoder(autoencoder), data_name, feature_name, newfamily, input_shape, 'cade').autoencoder
    elif sampler_name == 'transcendent':
        # pass the classifier architecture
        detector_model = copy_compiled_model(classifier, copy_weights=False)
    elif sampler_name == 'uncertainty' or sampler_name == 'entropy':
        detector_model = classifier

    model = adapt_classifier_arch(dream_model, num_family, lr=lr) 

    result_folder = os.path.join('results', data_name, feature_name.capitalize(), f'_bud_{budget} (tuningFalse_retrain{retrain_epoch}_aeFalse)')
    model_path = os.path.join(result_folder, f'{newfamily}_updated_{detector_name}.h5')
    f1, accuracy = active_learning_with_budget(model, x_train, y_train, x_test, y_test, 
                                               y_concept=y_concept, budget=budget, 
                                               verbose=0, retrain_epoch=retrain_epoch, 
                                               batch_size=args.batch_size, 
                                               old_model=detector_model, 
                                               model_path=model_path, num_family=num_family, 
                                               uncertainty_sampler=uncertainty_sampler, 
                                               lazy_run=False)

    consistent_rng_handler(x_train, y_train, x_test, y_test)
    del classifier, autoencoder, model, uncertainty_sampler, detector_model
    clear_models()
    gc.collect()
    
    return [newfamily, detector_name, f1, accuracy]


def retrain_with_autoencoder(data_name, feature_name, newfamily, num_family, lr, budget, retrain_epoch=50, ploss=True): 
    """Extended HCC for arbitrary (black-box) models and inter-class drift"""
    _, _, _, _, classifier = load_data_model(
        data_name, feature_name, model_type='basic', newfamily=newfamily, num_family=num_family, behavior_flag=False)
    x_train, y_train, x_test, y_test, autoencoder = load_data_model(
        data_name, feature_name, model_type='cade', newfamily=newfamily, num_family=num_family, behavior_flag=False)
    
    input_shape = autoencoder._input_shape
    summarize_classifier(classifier, input_shape)
    summarize_classifier(autoencoder, input_shape)

    classifier = load_trained_model(classifier, data_name, feature_name, newfamily, input_shape)  
    cade = load_trained_model(ContrastiveAutoencoder(autoencoder), data_name, feature_name, newfamily, input_shape, 'cade')
    if feature_name == 'drebin':
        lambda_sep, margin = .1, 10.
    elif feature_name == 'mamadroid':
        lambda_sep, margin = .1, 5.
    elif feature_name == 'damd':
        lambda_sep, margin = 1e-4, 10.
    stack_model = StackModel(classifier, cade.autoencoder, lambda_sep, margin)
    stack_model = adapt_classifier_arch(stack_model, num_family, lr=lr, stack=True)
    if ploss:
        uncertainty_sampler = DriftDetector('hcc').drift_sampler
        detector_model = stack_model
    else:
        uncertainty_sampler = DriftDetector('cade').drift_sampler
        detector_model = cade.autoencoder

    result_folder = os.path.join('results', data_name, feature_name.capitalize(), f'_bud_{budget} (tuningFalse_retrain{retrain_epoch}_aeFalse)')
    model_path = os.path.join(result_folder, f'{newfamily}_updated_{detector_name}.h5')
    f1, accuracy = active_learning_with_budget(stack_model, x_train, y_train, x_test, y_test, budget=budget, 
                                               verbose=0, retrain_epoch=retrain_epoch, 
                                               batch_size=args.batch_size, 
                                               old_model=detector_model, 
                                               model_path=model_path, num_family=num_family, 
                                               uncertainty_sampler=uncertainty_sampler, 
                                               lazy_run=False)

    return [newfamily, detector_name, f1, accuracy]


class StackModel(ContraBase):
    def __init__(self, classifier, autoencoder, lambda_sep, margin, binary_task=False):
        super().__init__()
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.lambda_sep = lambda_sep
        self.margin = margin
        self.binary_task = binary_task

    def calculate_loss(self, inputs, family_labels, training=True, **kwargs):
        embedding = self.autoencoder.encoder(inputs, training=training)
        rec_inputs = self.autoencoder.decoder(embedding, training=training)
        original_preds = self.classifier(inputs, training=training)
        rec_preds = self.classifier(rec_inputs, training=training)
        reduction = 'auto' if training else 'none'
        # Compute the losses
        self.cls_label = self.multi2binary_malware_label_tensor(family_labels) if self.binary_task else family_labels
        cls_loss = self.classification_loss(original_preds, reduction=reduction) 
        cls_loss_rec = self.classification_loss(rec_preds, reduction=reduction) 
        sep_loss = self.concept_separation_loss(embedding, family_labels, self.binary_task)
        if training:
            return cls_loss + cls_loss_rec, sep_loss
        else:
            return cls_loss[0] + self.lambda_sep * sep_loss

    def train_step(self, data):
        inputs, classifier_labels = data
        with tf.GradientTape() as tape:
            cls_loss, sep_loss = self.calculate_loss(inputs, classifier_labels)
            total_loss = cls_loss + self.lambda_sep * sep_loss
        trainable_vars = self.classifier.trainable_variables + self.autoencoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(self.cls_label, self(inputs))
        # return {'cls_loss': cls_loss, 'sep_loss': sep_loss, 'loss': total_loss}
        return {m.name: m.result() for m in self.metrics}
        
    def call(self, inputs, training=False):
        return self.classifier(inputs, training=training)

    def get_drift_scores(self, X_test, X_train, y_train, batch_size):
        hcc_sampler = PseudoBatch(self.autoencoder, self.classifier)
        hcc_sampler.setup_data(X_test, X_train, y_train, batch_size=8)
        anomaly_scores = []
        for idx in range(len(X_test)):
            x_batch, y_batch = hcc_sampler.form_hcc_pseudo_batch(idx, batch_size)
            pseudo_loss = self.calculate_loss(x_batch, y_batch, False)
            anomaly_scores.append(pseudo_loss)
        return np.array(anomaly_scores)


def consistent_rng_handler(x_train, y_train, x_test, y_test):
    for i in range(3): # print('random handler', i)
        form_tf_data((x_test[:2], y_test[:2]), batch_size=1) 
    sample_evenly(x_train, y_train, num_sample=budget)


def analyze_updated_results(df, key_column='classifier_tag', value_columns=['f1', 'accuracy'], log_info='', specific_key=None):
    def get_key_avgs(df, value, value_column):
        return df.loc[df[key_column] == value, value_column].mean()
    
    analysis_dict = {}
    ordered_list = ['transcendent', 'cade', 'vanilla', 'dream-', 'transcendent+', 'cade+', 'uncertainty+', 'dream'] # 'hcc', 'hcc+'
    existing_keys = df[key_column].unique()
    for value in [item for item in ordered_list if item in existing_keys]: 
        _avgs = {}
        for value_column in value_columns:
            _avgs[value_column] = get_key_avgs(df, value, value_column)
        analysis_dict[value] = _avgs
    analysis_df = pd.DataFrame(analysis_dict)
    if specific_key: analysis_df = analysis_df[['vanilla', specific_key, 'dream']]
    logging.info(f"Updated Results{log_info}:\n{analysis_df}")


def update_or_append(result_prime, new_df):
    index_columns = result_prime.columns[:2].tolist()
    # Set the first two columns as a MultiIndex
    result_prime.set_index(index_columns, inplace=True)
    new_df.set_index(index_columns, inplace=True)

    for index, row in new_df.iterrows():
        if index in result_prime.index: # Update existing rows
            result_prime.loc[index] = row
        else: # Append new rows
            result_prime = pd.concat([result_prime, row.to_frame().T])

    # Reset the index while preserving the original column names
    result_prime.reset_index(inplace=True)
    result_prime.columns = index_columns + result_prime.columns[len(index_columns):].tolist()

    return result_prime


if __name__ == '__main__':

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler(log_file, mode=file_mode),
            logging.StreamHandler()
        ] 
    )
    # detector_name = 'transcendent' # 'entropy' # 'cade' # 'uncertainty'
    data_name = args.data_name
    feature_name = args.feature_name
    dataset_families = get_dataset_family(data_name)
    num_family = len(dataset_families)
    lr = .0001
    retrain_epoch = 50
    
    if args.display_mode:
        for _budget in [10, 20, 30, 40, 100]:
            print(f'=== {_budget} ===')
            file_name = os.path.join('results', data_name, feature_name.capitalize(), f'active_bud_{_budget} (tuningFalse_retrain{retrain_epoch}_aeFalse).csv')
            results = pd.read_csv(file_name)
            analyze_updated_results(results)
        exit()

    budget = args.budget
    file_name = os.path.join('results', data_name, feature_name.capitalize(), f'active_bud_{budget} (tuningFalse_retrain{retrain_epoch}_aeFalse).csv')
    result_prime = pd.read_csv(file_name)

    detector_name = args.sampler_name   
    force_run = True
    tag = f'(Dataset-{data_name}, Feature-{feature_name}) Sampler-{detector_name} @Budget-{budget}'
    if not force_run and result_prime[result_prime.columns[1]].eq(detector_name).any():
        logging.debug(f'Results exist for {tag}, set global `force_run` to True if you still want to re-run.')
        analyze_updated_results(result_prime, log_info=f' exist for {tag}', specific_key=detector_name)
        exit()
    else: logging.info(f'Running {tag}')

    results = []
    for newfamily in range(num_family):
        logging.info(f"Hodling out on family {newfamily}-{dataset_families[newfamily]}")
        if detector_name == 'hcc':
            results.append(
                retrain_with_autoencoder(data_name, feature_name, newfamily, num_family, lr, budget, retrain_epoch)
            )
        elif detector_name == 'hcc+':
            results.append(
                retrain_with_autoencoder(data_name, feature_name, newfamily, num_family, lr, budget, retrain_epoch, ploss=False)
            )
        elif detector_name.endswith('+'):
            results.append(
                retrain_concept_with_base_sampler(detector_name, data_name, feature_name, newfamily, num_family, lr, budget, retrain_epoch)
            )
        else:
            results.append(
                retrain_with_base_sampler(detector_name, data_name, feature_name, newfamily, num_family, lr, budget, retrain_epoch)
            )

    new_df = pd.DataFrame(results, columns=result_prime.columns)
    result_update = update_or_append(result_prime, new_df)
    analyze_updated_results(result_update, log_info=f' for {tag}')
    result_update.to_csv(file_name, index=False)
