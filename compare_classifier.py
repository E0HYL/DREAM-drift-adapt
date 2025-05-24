#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   compare_classifier.py
@Time    :   2023/08/24 11:23:57
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import os
# TF INFO messages are not logging.infoed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import pandas as pd
import re
from glob import glob
import argparse
parser = argparse.ArgumentParser(description="Concept Drift Adaptation")
# Set up the command-line arguments
parser.add_argument("--data_name", default="drebin", type=str, help="Data name")
parser.add_argument("--feature_name", default="drebin", type=str, help="Feature name (options: drebin, damd, mamadroid)")
parser.add_argument("--retrain_epoch", default=50, type=int, help="Training epochs")
parser.add_argument("--lr", default=.0001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=32, type=int, help="Training batch size")
parser.add_argument("--al_setting", default="budget", type=str, help="Active learning setting (options: budget, threshold)")
parser.add_argument("--budget", default=50, type=int, help="Sample budget (only applicable when al_setting=`budget`)")
parser.add_argument("--threshold", default=0.9, type=float, help="Retrain threshold (only applicable when al_setting=`threshold`)")
parser.add_argument('--tuning', dest='tuning_flag', action='store_true', help='Set the flag for fine-tuning to True')
parser.add_argument('--ae_trainable', dest='trainable_flag', action='store_true', help='Set the flag for trainable concept autoencoder to True')
parser.add_argument('--force', dest='force_run', action='store_true', help='Re-run existing experiments')
parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID")
parser.add_argument('--display', dest='display_mode', action='store_true', help='Display all results of current data and feature')
parser.set_defaults(trainable_flag=False, tuning_flag=False, display_mode=False, force_run=False)
args = parser.parse_args()
data_name = args.data_name
feature_name = args.feature_name # drebin damd mamadroid
result_folder = os.path.join('results', data_name, feature_name.capitalize())
al_setting = args.al_setting
CONCEPT_CLS = data_name == 'drebin'
from test import set_gpu_id, lst_to_str 
set_gpu_id(args.gpu_id)
import logging
bud_thr_value = args.budget if args.al_setting == 'budget' else args.threshold
if args.display_mode:
    log_file, file_mode = os.path.join(result_folder, \
        f'al{al_setting.capitalize()}-{data_name.capitalize()}-{feature_name.capitalize()}.log'), 'w'  
else:
    log_file, file_mode = os.path.join(result_folder, \
        f'al{al_setting.capitalize()}-{bud_thr_value}-{args.retrain_epoch}-{data_name}-{feature_name}.log'), 'a'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode=file_mode),
        logging.StreamHandler()
    ] 
)
# Disable Propagation for Submodule
dream_load_data_logger = logging.getLogger('dream.load_data')
dream_load_data_logger.propagate = False
from dream import set_random_seed, set_gpu_growing
set_gpu_growing()
set_random_seed(0)
from dream.concept_learning import DREAM, recover_classifier
from dream.load_data import load_data_with_hold_out, load_trained_model, get_dataset_family
from dream.active_learning import active_learning_with_threshold, active_learning_with_budget, sample_dream_uncertainty, sample_uncertain_instances
from dream.classifiers import DAMDModel, DrebinMLP, MarkovMLP, summarize_classifier, modify_output, clone_copy_model
from dream.autoencoder import TabularAutoencoder, Conv1DTextAutoencoder
from dream.train_utils import default_classifier_compile, clear_models, ModelConfig 


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


def get_dream_model(feature_name, num_family, **kwargs):
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
    dream_config = ModelConfig().get_config(data_name, feature_name, 'dream')
    dream_model = DREAM(classifier, autoencoder.encoder, autoencoder.decoder, concept_cls=CONCEPT_CLS, **dream_config)
    return dream_model


def load_data_with_trained_model(data_name, feature_name, newfamily, num_family, behavior_flag=True, model_tags=None):
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

    model_tags = ['vanilla', 'dream-', 'dream'] if model_tags is None else model_tags
    if 'vanilla' in model_tags:
        classifier = get_classifier(feature_name, num_family, **kwargs)
        classifier = load_trained_model(classifier, data_name, feature_name, newfamily, input_shape)
    else:
        classifier = None
    if 'dream' in model_tags:
        dream_model = get_dream_model(feature_name, num_family, **kwargs)
        dream_model = load_trained_model(dream_model, data_name, feature_name, newfamily, input_shape, 'dream')
    else:
        dream_model = None
    if 'dream-' in model_tags:
        _dream_model = get_dream_model(feature_name, num_family, **kwargs)
        _dream_model = load_trained_model(_dream_model, data_name, feature_name, newfamily, input_shape, 'dream')
        dream_classifier = recover_classifier(_dream_model.f0, _dream_model.f1)
    else:
        dream_classifier = None

    return x_train, y_train, x_test, y_test, classifier, dream_classifier, dream_model


def adapt_classifier(model, num_classes, train_data=None, **kwargs):
    _freeze = False if train_data is None else True
    if model.name == 'dream':
        model.f1 = modify_output(model.f1, num_classes, _freeze, name=model.f1.name)
    else:
        model = modify_output(model, num_classes, _freeze, name=model.name)
    # Freeze customized model before compilation
    if model.name == 'dream':
        model.encoder.trainable = args.trainable_flag
        model.decoder.trainable = args.trainable_flag
    compile_loss = kwargs.get('loss', 'sparse_categorical_crossentropy')
    compile_metrics = kwargs.get('metric', ['sparse_categorical_accuracy'])
    compile_lr = kwargs.get('lr', .0004)
    model = default_classifier_compile(model, loss=compile_loss, metrics=compile_metrics, lr=compile_lr, debug=True)
    # if _freeze:
    #     num_epochs = kwargs.get('epochs', 20)
    #     logging.info(f'Fine-tuning with training data for {num_epochs} epochs')
    #     model.fit(*train_data, epochs=num_epochs, verbose=0)

    return model


def analyze_result(df, id_col='classifier_tag', target_id='vanilla', target_val=None, weights=None, maximizing=True):
    target_val = df.columns[2:] if target_val is None else target_val
    select_val = df[df[id_col]==target_id][target_val]
    if weights is None:
        final_val = select_val.mean().to_dict()
    else:
        weights = weights if maximizing else \
            np.exp(1-weights) / np.sum(np.exp(1-weights)) # softmax for inverted weights: the higher values become smaller while ensuring that the adjusted weights still sum up to 1
        final_val = weights.dot(select_val)
        final_val = {target_val[i]: final_val[i] for i in range(len(target_val))}
    return final_val


def generate_cmp_results(df, old_id='vanilla', new_id='dream', **kwargs):
    try:
        van_result = pd.Series(analyze_result(df, target_id=old_id, **kwargs))
        dre_result = pd.Series(analyze_result(df, target_id=new_id, **kwargs))
        maximizing = kwargs.get('maximizing', True)
        cmp_result = (dre_result - van_result) / van_result if maximizing \
                    else (van_result - dre_result) / van_result
        cmp_result = cmp_result.apply(lambda x: '{:.2%}'.format(x))
        df = pd.DataFrame({'w/o': van_result, 'w/': dre_result, 'Improves': cmp_result})  
        return df
    except Exception as e:
        logging.warning(e)
        return None


def run_al_threhold(threshold, thr_path, lazy_run=True):
    if lazy_run and os.path.exists(thr_path):
        thr_results = pd.read_csv(thr_path)
    else:
        logging.info(f'Comparing models trained on {data_name.upper()} dataset and {feature_name.upper()} feature.')
        thr_results = []
        for newfamily in range(num_family):
            logging.info(f"Hodling out on family {newfamily}-{dataset_families[newfamily]}")
            x_train, y_train, x_test, y_test, classifier, dream_model = load_data_with_trained_model(data_name, feature_name, newfamily, num_family)
            dream_classifier = recover_classifier(dream_model.f0, dream_model.f1)
            for tag, model in ({'vanilla': classifier, 'dream': dream_classifier}).items():
                logging.info(f'Active learning for [{tag}] classifier')
                model = modify_output(model, num_family)
                model = default_classifier_compile(model, loss='sparse_categorical_crossentropy', lr=.0004)
                model.fit(x_train, y_train, epochs=30, verbose=0)
                retrain_times, num_samples = active_learning_with_threshold(model, x_train, y_train, x_test, y_test, Th=threshold, verbose=0, retrain_epoch=args.retrain_epoch, batch_size=args.batch_size)
                thr_results.append([newfamily, tag, retrain_times, num_samples])
        thr_results = pd.DataFrame(thr_results, columns=['hold_out_idx', 'classifier_tag', 'retrain_times', 'num_samples'])
        thr_results.to_csv(thr_path, index=False)
    cmp_result = generate_cmp_results(thr_results, maximizing=False)
    logging.info(f'Active learning with threshold [{threshold}]: \n{cmp_result}')
    return cmp_result


def run_al_budget(budget, bud_path, lazy_run=True, model_tags=None):
    if lazy_run and os.path.exists(bud_path):
        bud_results = pd.read_csv(bud_path)
    else:
        logging.info(f'Comparing models trained on {data_name.upper()} dataset and {feature_name.upper()} feature.')
        bud_results = []
        intermediate_folder = bud_path.replace('/active_bud_', '/_bud_')[:-4]
        if not os.path.exists(intermediate_folder):
            os.makedirs(intermediate_folder)
        for newfamily in range(num_family):
            logging.info(f"Hodling out on family {newfamily}-{dataset_families[newfamily]}")
            x_train, y_train, x_test, y_test, classifier, dream_classifier, dream_model = load_data_with_trained_model(data_name, feature_name, newfamily, num_family, model_tags=model_tags)
            _y_train = y_train[0]
            _y_test = y_test[0]
            y_concept = (y_train[1], y_test[1])
            _cmp_dict = {'vanilla': classifier, 'dream-': dream_classifier, 'dream': dream_model}
            cmp_dict =  {key: _cmp_dict[key] for key in model_tags} if model_tags is not None else _cmp_dict
            for tag, model in cmp_dict.items(): 
                logging.info(f'Active learning for [{tag}] classifier')
                if tag == 'dream':
                    tuning_data = (x_train, (_y_train, y_concept[0])) if args.tuning_flag else None
                    dream_config = dream_model.get_config()
                    _y_concept, old_model = y_concept, clone_copy_model(model, classifier=None, encoder=dream_model.encoder, decoder=dream_model.decoder, f0=dream_model.f0, f1=dream_model.f1, concept_cls=CONCEPT_CLS, **dream_config) # recover_classifier(dream_model.f0, dream_model.f1)
                    uncertainty_sampler = sample_dream_uncertainty
                else:
                    tuning_data = (x_train, _y_train) if args.tuning_flag else None
                    _y_concept, old_model = None, model
                    uncertainty_sampler = sample_uncertain_instances
                model = adapt_classifier(model, num_family, train_data=tuning_data, lr=args.lr) 
                model_path = os.path.join(intermediate_folder, f'{newfamily}_updated_{tag}.h5')
                f1, accuracy = active_learning_with_budget(model, x_train, _y_train, x_test, _y_test, budget=budget, verbose=0, y_concept=_y_concept, retrain_epoch=args.retrain_epoch, batch_size=args.batch_size, old_model=old_model, model_path=model_path, num_family=num_family, uncertainty_sampler=uncertainty_sampler)
                bud_results.append([newfamily, tag, f1, accuracy])
                clear_models()
        bud_results = pd.DataFrame(bud_results, columns=['hold_out_idx', 'classifier_tag', 'f1', 'accuracy'])
        bud_path = bud_path if len(cmp_dict) == 3 else bud_path.replace('.csv', f"[{'+'.join(cmp_dict.keys())}].csv")
        bud_results.to_csv(bud_path, index=False)
    cmp_result = generate_cmp_results(bud_results)
    # logging.info(f"Active learning with budget [{budget}]: \n{generate_cmp_results(bud_results, new_id='dream-')}\n{cmp_result}\n") # 
    logging.info(f"Active learning with budget [{budget}]: \n{cmp_result}\n{generate_cmp_results(bud_results, old_id='dream-', new_id='dream')}")
    return cmp_result


def extract_variables_from_path(target_path, verbose=False):
    variables = {}
    file_name = os.path.basename(target_path)
    if al_setting == 'budget':
        budget_match = re.search(r'active_bud_([\d\.]+)', file_name)
        variables['budget'] = int(budget_match.group(1))
    elif al_setting == 'threshold':
        budget_match = re.search(r'active_thr_([\d\.]+)', file_name)
        variables['threshold'] = float(budget_match.group(1))
    retrain_match = re.search(r'retrain([\d\.]+)', file_name)
    variables['retrain_epoch'] = int(retrain_match.group(1))
    if verbose:
        trainable_match = re.search(r'ae(True|False)', file_name)
        variables['trainable_flag'] = True if trainable_match.group(1) == 'True' else False
        tuning_match = re.search(r'tuning(True|False)', file_name)
        variables['tuning_flag'] = True if tuning_match.group(1) == 'True' else False
    return variables


def display_results_with_setting(result_path=result_folder, metric='f1'):
    metric_idx = 0 if metric == 'f1' else 1
    setting_results = []
    for f in glob(os.path.join(result_path, 'active_*.csv')):
        settings = extract_variables_from_path(f)
        results = pd.read_csv(f)
        cmp_vanilla_dream = generate_cmp_results(results).iloc[metric_idx]
        cmp_vanilla_dream_minus = dict(generate_cmp_results(results, new_id='dream-').iloc[metric_idx])
        cmp_dream_minus_dream = dict(generate_cmp_results(results, old_id='dream-').iloc[metric_idx])

        cmp_vanilla_dream.index = ['vanilla', 'dream', 'improves(0)']
        settings.update(dict(cmp_vanilla_dream))
        settings['dream-'] = cmp_dream_minus_dream['w/o']
        settings['improves(1)'] = cmp_dream_minus_dream['Improves']
        settings['improves(-)'] = cmp_vanilla_dream_minus['Improves']
        setting_results.append(settings)
    return pd.DataFrame(setting_results).sort_values(by=['budget', 'retrain_epoch']).reset_index(drop=True)


if __name__ == '__main__':

    if args.display_mode:
        f1_results = display_results_with_setting(metric='f1')
        logging.info(f'=== F1-score Comparison ===: \n{f1_results}')
        acc_results = display_results_with_setting(metric='accuracy')
        logging.info(f'=== Accuracy Comparison ===: \n{acc_results}')
        exit()

    model_tags = ['vanilla', 'dream-', 'dream'] #  
    logging.info(f'Running drift adaptation for {lst_to_str(model_tags)} with args {vars(args)}.') 
    print(f'Results will be saved to {result_folder}.')
    dataset_families = get_dataset_family(data_name)
    num_family = len(dataset_families) # in the whole dataset

    lazy_run = not args.force_run
    if al_setting == 'threshold':
        # active learning with threshold
        threshold = args.threshold
        thr_path = os.path.join(result_folder, f'active_thr_{threshold} (tuning{args.tuning_flag}_retrain{args.retrain_epoch}_ae{args.trainable_flag}).csv')
        thr_cmp_result = run_al_threhold(threshold, thr_path, lazy_run) 
    elif al_setting == 'budget':
        # active learning with budget
        budget = args.budget
        bud_path = os.path.join(result_folder, f'active_bud_{budget} (tuning{args.tuning_flag}_retrain{args.retrain_epoch}_ae{args.trainable_flag}).csv')
        bud_cmp_result = run_al_budget(budget, bud_path, lazy_run, model_tags=model_tags)
