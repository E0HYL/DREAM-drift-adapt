#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/26 20:06:16
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Concept Drift Detection")
# Set up the command-line arguments
parser.add_argument("--data_name", default="drebin", type=str, help="Data name")
parser.add_argument("--feature_name", default="drebin", type=str, help="Feature name (options: drebin, damd, mamadroid)")
parser.add_argument("--model_name", default="dream", type=str, help="Model name (options: basic, cade, transcendent, dream)")
parser.add_argument("--num_epochs", default=250, type=int, help="Training epochs")
parser.add_argument("--batch_size", default=32, type=int, help="Training batch size")
parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID")
parser.add_argument("--bp_family", default=0, type=int, help="Training model from breakpoint of the hold-out family")
args = parser.parse_args()
from test import set_gpu_id
set_gpu_id(args.gpu_id)
import logging
from datetime import datetime
from dream import set_gpu_growing
set_gpu_growing()
from dream.load_data import load_data_model, get_dataset_family, load_trained_model, get_model_path
from dream.classifiers import summarize_classifier
from dream.autoencoder import Autoencoder
from dream.train_utils import train_model, default_classifier_compile
from dream.baseline_cade import epoch_train, detect_drift
from dream.baseline_transcend import get_transcend_scores
from dream.concept_learning import train_dream_model
from dream.evaluate import evaluate_detection, draw_multiple_roc_curves, evaluate_uncertainty_detection, evaluate_intra_drift_robustness, evaluate_concept_accuracy


def train_basic_model(data_name='drebin', feature_name='damd', newfamily=0, num_family=8, split_year=2015, num_epochs=30, eval_k=100, use_entropy=True, test_small_family=False, **kwargs):
    x_train, y_train, x_test, y_test, classifier = load_data_model(data_name=data_name, feature_name=feature_name, newfamily=newfamily, num_family=num_family, split_year=split_year, behavior_label=False)

    summarize_classifier(classifier, x_train.shape[1:])
    num_classes_train = classifier.layers[-1].output_shape[-1]
    loss = 'binary_crossentropy' if num_classes_train == 2 else 'categorical_crossentropy'
    model_tag = split_year if data_name == 'binarycls' else newfamily
    model_path = get_model_path(data_name, feature_name, model_tag, 'basic')
    # if os.path.exists(model_path): # continue training
    #     classifier.load_weights(model_path)
    y_train_ = np.eye(num_classes_train)[y_train]
    model = train_model(x_train, y_train_, classifier, loss=loss, model_path=model_path, num_epochs=num_epochs, lr=.001, **kwargs)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test[y_test!=num_family-1])
    train_acc = (y_train_pred.argmax(-1)==y_train).sum() / len(y_train)
    test_acc = (y_test_pred.argmax(-1)==y_test[y_test!=num_family-1]).sum() / len(y_test[y_test!=num_family-1])
    print('Train Acc', train_acc, 'Test Acc', test_acc)
    
    if data_name == 'binarycls':
        train_data = (x_train, y_train) if eval_with_train_data else None
        return evaluate_intra_drift_robustness(model, x_test, y_test, train_data=train_data, split_year=split_year)
    if test_small_family and data_name=='malradar':
        x_test_small = np.load(f'data/malradar/{feature_name.capitalize()}/small_test_families.npy', allow_pickle=True)
        x_test = np.concatenate([x_test, x_test_small])
        y_small_test = [num_family-1 for _ in range(len(x_test_small))]
        y_test = np.concatenate([y_test, y_small_test]) 
    return evaluate_inter_drift_detection(model, x_test, y_test, num_classes_train, eval_k, newfamily, use_entropy=use_entropy)


def train_transcendent_model(data_name='drebin', feature_name='damd', newfamily=0, num_family=8, split_year=2015, num_epochs=30, eval_k=100, test_small_family=False, **kwargs):
    x_train, y_train, x_test, y_test, classifier = load_data_model(data_name=data_name, feature_name=feature_name, newfamily=newfamily, num_family=num_family, split_year=split_year, behavior_label=False)

    summarize_classifier(classifier, x_train.shape[1:])
    num_classes_train = classifier.layers[-1].output_shape[-1]
    compile_loss = kwargs.get('loss', 'sparse_categorical_crossentropy')
    compile_metrics = kwargs.get('metric', ['sparse_categorical_accuracy'])
    compile_lr = kwargs.get('lr', .0004)
    classifier = default_classifier_compile(classifier, loss=compile_loss, metrics=compile_metrics, lr=compile_lr, debug=True)
    model_tag = split_year if data_name == 'binarycls' else newfamily
    model_path = get_model_path(data_name, feature_name, model_tag, 'transcendent')
    
    if test_small_family and data_name=='malradar':
        x_test_small = np.load(f'data/malradar/{feature_name.capitalize()}/small_test_families.npy', allow_pickle=True)
        x_test = np.concatenate([x_test, x_test_small])
        y_small_test = [num_family-1 for _ in range(len(x_test_small))]
        y_test = np.concatenate([y_test, y_small_test])    
    anomaly_scores = get_transcend_scores(x_train, y_train, x_test, classifier, num_epochs, model_path)

    auc_score, accuracy, (y_true, y_pred) = evaluate_detection(anomaly_scores, y_test, num_classes_train, eval_k, verbose=True)
    logging.info(f'Drift Detection (Hold-out # {newfamily}) - AUC: {round(auc_score, 4)}, Accuracy: {round(accuracy, 4)}')    
    return y_true, y_pred, accuracy  


def train_cade_ae(data_name='drebin', feature_name='damd', newfamily=0, batch_size=16, num_epochs=50, drift_class=7, eval_k=100, test_small_family=False, **kwargs):
    x_train, y_train, x_test, y_test, autoencoder = load_data_model(data_name=data_name, feature_name=feature_name, newfamily=newfamily, num_family=num_family, model_type='cade', behavior_label=False)
    autoencoder.build()
    input_shape = autoencoder._input_shape
    # logging.debug(f'Building CADE with encoder and decoder structures: \n{str(autoencoder.encoder.summary())}\n{str(autoencoder.decoder.summary())}')
    model_path = get_model_path(data_name, feature_name, newfamily, 'cade')
    trained_model = epoch_train(x_train=x_train, y_train=y_train, epochs=num_epochs, batch_size=batch_size,input_shape=input_shape, \
                                autoencoder=autoencoder, _model_path=model_path, **kwargs)
    if test_small_family and data_name=='malradar':
        x_test_small = np.load(f'data/malradar/{feature_name.capitalize()}/small_test_families.npy', allow_pickle=True)
        x_test = np.concatenate([x_test, x_test_small])
        y_small_test = [drift_class for _ in range(len(x_test_small))]
        y_test = np.concatenate([y_test, y_small_test])
    return evaluate_inter_drift_detection(trained_model.autoencoder, x_test, y_test, drift_class, eval_k, newfamily, train_data=(x_train, y_train))


def train_baseline(data_name='drebin', feature_name='damd', newfamily=0, num_epochs=30):
    # x_train, y_train, x_test, y_test, original_model, train_metrics = train_basic_model(data_name=data_name, feature_name=feature_name, newfamily=newfamily, num_epochs=0)
    breakpoint()


def dream_train(data_name='drebin', feature_name='damd', newfamily=0, split_year=2015, drift_class=7, num_epochs=30, cl_weights=True, cl_trainable=True, eval_with_train_data=False, detect_with_reliability=False, eval_k=100, hierarchical=False, output_class=7, test_small_family=False, **kwargs):
    del_emb = False
    x_train, y_train, x_test, y_test, (classifier, autoencoder) = load_data_model(data_name=data_name, feature_name=feature_name, newfamily=newfamily, split_year=split_year, model_type='dream', num_family=num_family, behavior_flag=True, emb_layer=not del_emb)
    input_shape =  x_train.shape[1:]
    summarize_classifier(classifier, input_shape)
    if feature_name == 'damd' and del_emb:
        split_layer_idx = 1 
        ae_input_shape = classifier.layers[0].output_shape[1:]
    else:
        split_layer_idx = 0
        ae_input_shape = input_shape
    summarize_classifier(autoencoder, ae_input_shape)
    model_tag = split_year if data_name == 'binarycls' else newfamily
    model_path, config_path = get_model_path(data_name, feature_name, model_tag, 'dream', config=True)
    if num_epochs > 0:
        if cl_weights:
            classifier = load_trained_model(classifier, data_name, feature_name, model_tag, input_shape)
            classifier.trainable = cl_trainable 
    text_input_flag = autoencoder.data_type == 'text'
    best_model = train_dream_model(x_train, y_train[0], y_train[1], classifier, autoencoder.encoder, autoencoder.decoder, model_path, config_path=config_path, num_epochs=num_epochs, text_input=text_input_flag, split_layer_idx=split_layer_idx, hierarchical=hierarchical, output_class=output_class, **kwargs)

    train_data = (x_train, y_train[0]) if eval_with_train_data else None
    if data_name == 'binarycls':
        return evaluate_intra_drift_robustness(best_model, x_test, y_test[0], train_data=train_data, split_year=split_year)
    
    # concept_accuracy = evaluate_concept_accuracy(best_model, x_test[y_test[0]!=drift_class], y_test[1][y_test[0]!=drift_class], batch_size=kwargs.get('batch_size'))
    # logging.info(f'Concept accuracy on ID test data: {concept_accuracy}')
    if eval_with_train_data:
        dream_ae = Autoencoder(autoencoder.data_type, autoencoder._input_shape)
        dream_ae.build_ae(best_model.encoder, best_model.decoder)
        _best_model = dream_ae
    else:
        _best_model = best_model
    detection_method = 'tloss' if detect_with_reliability else 'basic'
    if test_small_family and data_name=='malradar':
        x_test_small = np.load(f'data/malradar/{feature_name.capitalize()}/small_test_families.npy', allow_pickle=True)
        x_test = np.concatenate([x_test, x_test_small])
        y_small_test = [drift_class for _ in range(len(x_test_small))]
        y_test_family = np.concatenate([y_test[0], y_small_test])
    else:
        y_test_family = y_test[0]
    return evaluate_inter_drift_detection(_best_model, x_test, y_test_family, drift_class, eval_k, newfamily, train_data, method=detection_method)


def evaluate_inter_drift_detection(best_model, x_test, y_test, drift_class, eval_k, newfamily, train_data=None, **kwargs):
    if train_data is not None:
        x_train, y_train = train_data
        anomaly_scores, drift_y_preds = detect_drift(x_train, y_train, x_test, best_model) # should be AutoEncoder
        auc_score, (accuracy, accuracy_thr), (y_true, y_pred) = evaluate_detection(anomaly_scores, y_test, drift_class, drift_y_preds=drift_y_preds, verbose=True, top_k=eval_k)
        logging.info(f'Drift Detection (Hold-out # {newfamily}) - AUC: {round(auc_score, 4)}, Accuracy-thr: {round(accuracy_thr, 4)}, Accuracy-top: {round(accuracy, 4)}')
    else:
        auc_score, accuracy, (y_true, y_pred) = evaluate_uncertainty_detection(best_model, x_test, y_test, drift_class, eval_k, **kwargs)
        logging.info(f'Drift Detection (Hold-out # {newfamily}) - AUC: {round(auc_score, 4)}, Accuracy: {round(accuracy, 4)}')    
    return y_true, y_pred, accuracy    


if __name__ == '__main__':
    
    data_name = args.data_name
    feature_name = args.feature_name
    model_name = args.model_name

    dataset_families = get_dataset_family(data_name)
    num_family = len(dataset_families) # in the whole dataset

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    log_stamp = datetime.now().strftime("%m%d%H%M%S")
    log_file = f"logs/{log_stamp}-{data_name}-{feature_name}-{model_name}.log" if num_epochs > 0 else f"logs/evaluation-{data_name}-{feature_name}-{model_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ] 
    )
    logging.info(f'Using {data_name.upper()} dataset and {feature_name.upper()} feature.')

    if data_name == 'binarycls':
        split_year = 2015
        eval_with_train_data = True
        if model_name == 'basic':
            model_aut = train_basic_model(data_name, feature_name, batch_size=batch_size, num_epochs=num_epochs, split_year=split_year)
        elif model_name == 'dream':
            lambda_rec, lambda_sep, lambda_rel, lambda_pre, margin = .1, .001, .001, .001, 10. # TODO: hyper-parameters for different feature analyzers
            hierarchical = True # TODO: maybe also compare with CADE_encoder
            output_class = 2
            model_aut = dream_train(data_name, feature_name, num_epochs=num_epochs, batch_size=batch_size, lambda_rec=lambda_rec, lambda_sep=lambda_sep, lambda_rel=lambda_rel, lambda_pre=lambda_pre, sep_margin=margin, hierarchical=hierarchical, output_class=output_class, split_year=split_year)
        logging.info(f"[{model_name}] model robustness evaluation: \n{model_aut}")
    else:
        y_true_lst = []
        y_pred_lst = []
        accuracy_list = []
        for i in range(args.bp_family, num_family):
            logging.info(f"Hodling out on family {i}-{dataset_families[i]}")
            if model_name == 'basic':
                use_entropy = False
                family_y_true, family_y_pred, family_acc = train_basic_model(data_name, feature_name=feature_name, newfamily=i, batch_size=batch_size, num_epochs=num_epochs, num_family=num_family, use_entropy=use_entropy)
            elif model_name == 'transcendent':
                family_y_true, family_y_pred, family_acc = train_transcendent_model(data_name, feature_name=feature_name, newfamily=i, batch_size=batch_size, num_epochs=num_epochs, num_family=num_family)
            elif model_name == 'cade':
                if feature_name == 'drebin':
                    lambda_, margin = .1, 10.
                elif feature_name == 'mamadroid':
                    lambda_, margin = .1, 5.
                elif feature_name == 'damd':
                    lambda_, margin = 1e-4, 10.
                family_y_true, family_y_pred, family_acc = train_cade_ae(data_name, feature_name=feature_name, newfamily=i, batch_size=batch_size, drift_class=num_family-1, num_epochs=num_epochs, lambda_=lambda_, margin=margin)
            elif model_name == 'base': # TODO: modify the output layer for behavior classification 
                family_y_true, family_y_pred, family_acc = train_baseline(data_name, feature_name, newfamily=i)
            elif model_name == 'dream':
                concept_cls = True
                cl_trainable, mon_metric = True, None 
                # cl_trainable, mon_metric = False, 'loss'
                if feature_name == 'drebin':
                    lambda_rec, lambda_sep, lambda_rel, lambda_pre, margin = .1, .01, .01, .001, 10.
                elif feature_name == 'mamadroid':
                    concept_cls = None
                    lambda_rec, lambda_sep, lambda_rel, lambda_pre, margin = .01, .001, .001, .001, 5. 
                elif feature_name == 'damd':
                    if data_name == "drebin":
                        lambda_rec, lambda_sep, lambda_rel, lambda_pre, margin = .01, .001, .001, .0001, 5. 
                    elif data_name == "malradar":
                        lambda_rec, lambda_sep, lambda_rel, lambda_pre, margin = .001, 1e-7, 1e-7, 1e-7, 10.
                train_data_available = False # option for evaluation in dream
                detect_with_reliability = True
                family_y_true, family_y_pred, family_acc = dream_train(data_name, feature_name, newfamily=i, drift_class=num_family-1, num_epochs=num_epochs, batch_size=batch_size, lambda_rec=lambda_rec, lambda_sep=lambda_sep, lambda_rel=lambda_rel, lambda_pre=lambda_pre, sep_margin=margin, concept_cls=concept_cls, cl_trainable=cl_trainable, mon_metric=mon_metric, eval_with_train_data=train_data_available, detect_with_reliability=detect_with_reliability)
            y_true_lst.append(family_y_true)
            y_pred_lst.append(family_y_pred)
            accuracy_list.append(family_acc)
        
        result_dir = os.path.join('results', data_name, feature_name.capitalize())
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        pic_name = f'{model_name}_mean_roc'
        np.savez(os.path.join(result_dir, pic_name), 
                 drift_scores=np.array(y_pred_lst, dtype='object'), 
                 drift_labels=np.array(y_true_lst, dtype='object'), 
                 allow_pickle=True)
        if model_name == 'dream':
            if train_data_available: pic_name += '_train'
            if detect_with_reliability: pic_name += '_rloss'
        elif data_name != 'binarycls' and model_name == 'basic':
            if use_entropy: 
                model_name = 'Entropy'
            else:
                model_name = 'Prob.'
                pic_name += '_prob'
        mean_roc_path = os.path.join(result_dir, pic_name+'.png') #_addsmall
        # dataset_families = [fam+'+Small' for fam in dataset_families]
        mean_auc, std_auc = draw_multiple_roc_curves(y_true_lst, y_pred_lst, dataset_families, figpath=mean_roc_path, draw_average=True, title_suffix=f' ({model_name.capitalize()} + {feature_name.capitalize()}) ')
        acc_arr = np.array(accuracy_list)
        avg_acc = f"{np.mean(acc_arr):.3f} ± {np.std(acc_arr):.3f}"
        with open(os.path.join(result_dir, 'top_acc.txt'), 'a') as f:
            f.write(f"{model_name}: {acc_arr} {avg_acc}" + '\n')
        logging.info(f"Mean ROC-AUC = {mean_auc:.3f} ± {std_auc:.3f}, Accuracy = {avg_acc}")
