#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   binary_main.py
@Time    :   2023/12/18 10:50:04
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
from test import set_gpu_id
import argparse
parser = argparse.ArgumentParser(description="Concept Drift Detection for Binary Classification")
parser.add_argument("--detector_name", default="cade", type=str, help="Detector name")
parser.add_argument("--num_epochs", default=200, type=int, help="Training epochs")
parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID")
args = parser.parse_args()
set_gpu_id(args.gpu_id)
import numpy as np
from datetime import datetime
import logging

from dream import set_gpu_growing
set_gpu_growing()
from dream.train_sampler import DistributedSampler, ContrastiveSampler, relabel_for_unique
from dream.classifiers import DrebinMLP
from dream.train_utils import train_model

from dream.load_data import load_data_with_year_split, load_trained_model
from dream.classifiers import DAMDModel, DrebinMLP, MarkovMLP, summarize_classifier, copy_compiled_model
from dream.autoencoder import TabularAutoencoder, Conv1DTextAutoencoder, Autoencoder
from dream.ood_detector import ContrastiveDetector, evaluate_detector_with_model_predictions
from dream.baseline_transcend import get_transcend_scores
from dream.train_utils import get_cls_pseudo_loss, default_classifier_compile, clear_models, ModelConfig
from dream.concept_learning import train_dream_model, recover_classifier, DREAM
from dream.evaluate import evaluate_intra_drift_robustness, multi2binary_malware_label, evaluate_model
from dream.active_learning import active_learning_with_budget, sample_uncertain_instances, sample_dream_uncertainty


def train_seperate_ood_detector(autoencoder, X_train, y_train, batch_size, detector_name='cade', margin=10.0, lambda_=1e-2, num_epochs=200, learning_rate=.0004, tag=2015, debug_mode=True, model_dir=None):
    if num_epochs > 0:
        logging.info(f'Training {detector_name}[margin={margin}, lambda={lambda_}] for {num_epochs} epochs with lr={learning_rate}')
    if detector_name == 'cade':
        hierarchical = False
        sampler = ContrastiveSampler(X_train, y_train, batch_size)
        # original ae: enc+dec
    elif detector_name == 'hcc':
        hierarchical = True
        sampler = DistributedSampler(X_train, y_train, batch_size)
        # enc+mlp
        mlp_dims = [autoencoder.encoding_dim, 100, 2]
        mlp = DrebinMLP(mlp_dims, name='hcc_mlp')
        autoencoder.decoder = mlp
    dataset = sampler.create_dataset()
    summarize_classifier(autoencoder, X_train.shape[1:], silent=False)
    detector = ContrastiveDetector(autoencoder, margin=margin, lambda_=lambda_, hierarchical=hierarchical)
    summarize_classifier(detector, X_train.shape[1:])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    detector = train_model(dataset, None, detector, batch_size=batch_size, num_epochs=num_epochs, model_path=f'{model_dir}/{tag}_{margin}_{lambda_}.h5', metrics=['total_loss'], mode='min', lr=learning_rate, debug=debug_mode, alternate_monitor='contrastive_loss', verbose_batch=False, steps_per_epoch=X_train.shape[0] // batch_size)

    return detector


def load_data_model(feature_name='drebin', model_type='basic', behavior_flag=False, split_year=2015):
    analyzer = feature_name.capitalize()
    family_flag = False if model_type == 'basic' else True
    x_train, y_train, x_test, y_test = load_data_with_year_split(split_year, analyzer=analyzer, behavior_label=behavior_flag, family_label=family_flag, end_year=2020)
    input_shape = x_train.shape[1:]
    
    assert model_type in ['basic', 'cade', 'hcc', 'dream', 'none']
    if feature_name == 'damd':
        vocab_size = 218 # max(x_train.max(), x_test.max()) + 1
        classifier = DAMDModel(no_tokens=vocab_size, no_labels=2)
        autoencoder = Conv1DTextAutoencoder(vocab_size=vocab_size, sequence_length=input_shape[-1])
    elif feature_name == 'mamadroid':
        classifier = MarkovMLP(dims=[1000, 200, 2])
        autoencoder = TabularAutoencoder(input_dim=np.multiply(*input_shape), hidden_dim=2048, encoding_dim=128)
        x_train = (x_train*10).reshape(len(x_train), -1)
        if type(x_test) == dict:
            for k in x_test:
                v = x_test[k]
                x_test[k] = (v*10).reshape(len(v), -1)
        else:
            x_test = (x_test*10).reshape(len(x_test), -1)
    elif feature_name == 'drebin':
        num_features = input_shape[-1]
        classifier = DrebinMLP(dims=[num_features, 100, 30, 2])
        autoencoder = TabularAutoencoder(input_dim=num_features, hidden_dim=512, encoding_dim=32)
    else:
        raise ValueError(f'no classifier named {feature_name}')
    if model_type == 'basic':
        model = classifier
    elif model_type == 'cade' or model_type == 'hcc':
        model = autoencoder
    elif model_type == 'dream' or model_type == 'none':
        model = (classifier, autoencoder)
    
    return x_train, y_train, x_test, y_test, model


def train_detector(feature_name='drebin', detector_name='cade', batch_size=64, num_epochs=200, learning_rate=.0004, split_year=2015, margin=10.0, lambda_=1e-2):
    model_dir = f'models/binarycls/{feature_name.capitalize()}/{detector_name}'
    set_log_file(model_dir)
    X_train, y_train, X_test, y_test, (classifier, detector) = load_data_model(feature_name=feature_name, model_type='none')
    if detector_name == 'cade' or detector_name == 'hcc':
        best_model = train_seperate_ood_detector(detector, X_train=X_train, y_train=y_train, batch_size=batch_size, detector_name=detector_name, num_epochs=num_epochs, learning_rate=learning_rate, tag=split_year, margin=margin, lambda_=lambda_, model_dir=model_dir)
    elif detector_name == 'transcendent':
        summarize_classifier(classifier, X_train.shape[1:])
        classifier_arch = copy_compiled_model(default_compile(classifier, learning_rate), False)
        get_transcend_scores(X_train, multi2binary_malware_label(y_train), None, classifier_arch, num_epochs, model_path=os.path.join(model_dir, f'{split_year}.h5'))

    classifier = load_trained_model(classifier, 'binarycls', feature_name, split_year, X_train.shape[1:])
    for year in X_test:
        test_data = X_test[year]
        test_label = y_test[year]
        test_pred = classifier.predict(test_data)
        if detector_name == 'cade' or detector_name == 'hcc':
            anomaly_scores = best_model.detect_drift(X_train, relabel_for_unique(y_train, hierarchical_check=True), test_data)
        elif detector_name == 'transcendent':
            anomaly_scores = get_transcend_scores(X_train, multi2binary_malware_label(y_train), test_data, classifier_arch, train_epoch=0, model_path=os.path.join(model_dir, f'{split_year}.h5'))
        elif detector_name == 'uncertainty':
            anomaly_scores = 1 - (test_pred).max(-1)
        elif detector_name == 'ploss':
            anomaly_scores = get_cls_pseudo_loss(test_pred)
        detection_score = evaluate_detector_with_model_predictions(test_pred, test_label, anomaly_scores)
        logging.info(f'{detector_name.upper()} # {year}: {detection_score}')


def get_model_path(feature_name, tag, model_type, config=False):
    model_folder = os.path.join('models/binarycls', feature_name.capitalize(), model_type)
    set_log_file(model_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder) 
    model_path = os.path.join(model_folder, f'{tag}.h5')
    if config:
        config_path = os.path.join(model_folder, f'model_config.txt')
        return model_path, config_path
    return model_path


def train_detector_with_concept(feature_name='drebin', batch_size=64, num_epochs=200, learning_rate=.0004, split_year=2015, hierarchical=True, cls_trainable=False, **kwargs):
    model_tag = 'dream'
    model_path, config_path = get_model_path(feature_name, split_year, model_tag, config=True)
    
    X_train, y_train, X_test, y_test, (classifier, autoencoder) = load_data_model(feature_name=feature_name, model_type='dream', behavior_flag=True)
    input_shape =  X_train.shape[1:]
    classifier = load_trained_model(classifier, 'binarycls', feature_name, split_year, input_shape)

    summarize_classifier(classifier, input_shape)
    summarize_classifier(autoencoder, input_shape)
    classifier.trainable = cls_trainable
    text_input_flag = autoencoder.data_type == 'text'

    if num_epochs > 0:
        logging.info(f'Training {model_tag} with lr={learning_rate}')
    best_model = train_dream_model(X_train, y_train[0], y_train[1], classifier, autoencoder.encoder, autoencoder.decoder, model_path, config_path=config_path, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs, text_input=text_input_flag, hierarchical=hierarchical, output_class=2,**kwargs)

    updated_classifier = recover_classifier(best_model.f0, best_model.f1) if cls_trainable else classifier
    detector_ae = Autoencoder(autoencoder.data_type, autoencoder._input_shape)
    detector_ae.build_ae(best_model.encoder, best_model.decoder)

    for year in X_test:
        test_data = X_test[year]
        test_label = y_test[0][year]
        test_pred = updated_classifier.predict(test_data)

        """Testing effects of different pseudo loss based samplers"""
        # anomaly_scores = []
        # 1. without training data
        # anomaly_scores = [best_model.get_dream_scores(test_data).T]
        # if autoencoder.trainable and num_epochs > 0:
        #     # 2. with training data
        #     detector = ContrastiveDetector(detector_ae, margin=best_model.margin, lambda_=best_model.lambda_sep, hierarchical=hierarchical)
        #     scores_with_training = detector.detect_drift(X_train, relabel_for_unique(y_train[0], hierarchical_check=True), test_data, classifier=updated_classifier)
        #     scores_with_training = scores_with_training if hierarchical else np.expand_dims(scores_with_training, 1)
        #     anomaly_scores.append(scores_with_training)
        # anomaly_scores = np.concatenate(anomaly_scores, 1)
        anomaly_scores = best_model.get_drift_scores(test_data)

        detection_score = evaluate_detector_with_model_predictions(test_pred, test_label, anomaly_scores)
        logging.info(f'{model_tag.upper()} # {year}: {detection_score}')


def adapt_trained_classifier(feature_name='drebin', model_tag='basic', split_year=2015, budget=50, repoch=50, compile_lr=.0001, lazy=False, **kwargs):
    data_name = 'binarycls'
    adapted_model_tag = model_tag+f'/adapted_budget-{budget}_repoch-{repoch}'
    adapted_path = get_model_path(feature_name, split_year, adapted_model_tag)

    if model_tag == 'basic':
        X_train, y_train, X_test, y_test, classifier = load_data_model(feature_name, model_tag)
        input_shape =  X_train.shape[1:]
        trained_model = load_trained_model(classifier, data_name, feature_name, split_year, input_shape)
        # default uncertainty sampler
        ood_model = None
        uncertainty_sampler = sample_uncertain_instances
    else:
        X_train, y_train, X_test, y_test, (classifier, detector) = load_data_model(feature_name, model_tag.split('_')[0], behavior_flag=True)
        input_shape =  X_train.shape[1:]
        summarize_classifier(classifier, input_shape)
        summarize_classifier(detector, input_shape)
        dream_config = ModelConfig().get_config(data_name, feature_name, model_tag)
        logging.info(f'DREAM config: {dream_config}')
        dream_model = DREAM(classifier, detector.encoder, detector.decoder, output_class=2, **dream_config)
        trained_model = load_trained_model(dream_model, data_name, feature_name, split_year, input_shape, model_name=model_tag)
        
        enc_trainable_flag = kwargs.pop('encoder_trainable', True)
        dec_trainable_flag = kwargs.pop('decoder_trainable', True)
        trained_model.encoder.trainable = enc_trainable_flag
        trained_model.decoder.trainable = dec_trainable_flag
        logging.info(f'Encoder trainable: {enc_trainable_flag}, Decoder trainable: {dec_trainable_flag}')

        ood_model = None
        uncertainty_sampler = sample_dream_uncertainty

    logging.info(f'lr = {compile_lr}')
    trained_model = default_compile(trained_model, compile_lr, debug=True)
        
    for year in X_test:
        test_data = X_test[year]
        if model_tag == 'basic':
            test_label = multi2binary_malware_label(y_test[year])
            y_concept = None
            _y_train = multi2binary_malware_label(y_train)
        else:
            test_label = y_test[0][year]
            y_concept = (y_train[1], y_test[1][year]) 
            _y_train = y_train[0]
        _model_path = adapted_path.replace('.h5', f'_{year}.h5')
        if os.path.exists(_model_path) and lazy:
            trained_model = load_trained_model(classifier, data_name, feature_name, f'{split_year}_{year}', input_shape,model_name=adapted_model_tag) if model_tag == 'basic' else\
                load_trained_model(dream_model, data_name, feature_name, f'{split_year}_{year}', input_shape, model_name=adapted_model_tag)
        else:
            active_learning_with_budget(trained_model, X_train, _y_train, test_data, test_label, y_concept=y_concept, budget=budget, retrain_epoch=repoch, num_family=None, verbose=0, model_path=_model_path, sample=True, old_model=ood_model, uncertainty_sampler=uncertainty_sampler, monitor='loss', mode='min', check_binary=True, **kwargs)
            clear_models()
        results = evaluate_model(trained_model, test_data, test_label, metric_name=['f1', 'acc', 'fpr', 'fnr'], legacy_format=False, check_binary=True)
        logging.info(f'{year}: {results}')


def default_compile(model, compile_lr, debug=False):
    return default_classifier_compile(model, lr=compile_lr, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], debug=debug)


def set_log_file(log_base):
    if not os.path.exists(log_base):
        os.makedirs(log_base)
    log_stamp = datetime.now().strftime("%m%d%H%M%S")
    log_file = f"{log_base}/{log_stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ] 
    )


if __name__ == '__main__':
    lambda_rec, lambda_sep, lambda_rel, lambda_pre, margin = 1., 1e-3, 1e-3, 1e-3, 10.
    
    """Exp 1: train the original ood detector seperately
    """
    # train_detector(detector_name=args.detector_name, num_epochs=args.num_epochs, margin=margin, lambda_=lambda_sep)
    
    """Exp 2: 1) train the ood detector with concept
    2) train the classifier and detector jointly with concept
    """
    # hierarchical = True
    # cls_trainable = False # DREAM(CADE) > CADE, DREAM(HCC) > HCC
    # if cls_trainable:
    #     detector_lr = 1e-4 if det_weights else 4e-4 
    #     learning_rate = {'detector_lr': detector_lr, 'classifier_lr': 1e-4}  
    #     lambda_rel = 1.
    # else:
    #     learning_rate = .0004
    # train_detector_with_concept(hierarchical=hierarchical, cls_trainable=cls_trainable, num_epochs=args.num_epochs, lambda_rec=lambda_rec, lambda_sep=lambda_sep, lambda_rel=lambda_rel, lambda_pre=lambda_pre, sep_margin=margin, learning_rate=learning_rate)

    budget = 100
    repoch = 100
    adapt_trained_classifier(model_tag='basic', budget=budget, repoch=repoch)
    # adapt_trained_classifier(model_tag='dream', budget=budget, decoder_trainable=True, repoch=repoch)   