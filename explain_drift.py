#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   explain_drift.py
@Time    :   2024/01/09 20:35:04
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2024
@Desc    :   None
'''

# here put the import lib
import os
import logging
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description="Concept Drift Explanation")
# Set up the command-line arguments
parser.add_argument("--data_name", default="drebin", type=str, help="Data name")
parser.add_argument("--feature_name", default="drebin", type=str, help="Feature name (options: drebin, damd, mamadroid)")
parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID")
args = parser.parse_args()
from tqdm import tqdm
from test import set_gpu_id
set_gpu_id(args.gpu_id)
from dream import set_gpu_growing
set_gpu_growing()
from dream.explain import explain_instance
from dream.load_data import get_dataset_family
from dream.data_utils import load_data_with_trained_model
from dream.baseline_cade import get_latent_data_for_each_family, get_latent_distance_between_sample_and_centroid, get_MAD_for_each_family
from dream.explain_utils import get_train_ncm, evaluate_drift_exp_credibility, generate_random_binary_masks


def main(data_name, feature_name, newfamily, num_family, model_name, exp_name, concept_space=False, **kwargs):
    x_train, (y_train, yb_train), x_test, (y_test, yb_test), dream_model = load_data_with_trained_model(data_name, feature_name, newfamily, num_family, dream=(model_name == 'dream'))
    exp_folder = os.path.join(f'_exp_{exp_name}', f'{data_name}-{feature_name}-{model_name}')
    if exp_name == 'dream':
        cert_loss_flag = True
        # exp_folder = os.path.join(exp_folder, f'lambda_unc_{kwargs.get("lambda_unc")}-lambda_con_{kwargs.get("lambda_con")}')
    else:
        cert_loss_flag = False
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    dis_exp_path = f"{exp_folder}/{newfamily}_drift_{'concept_' if concept_space else ''}exp.npy"
    family_behaviors = [yb_train[y_train == i][0] for i in range(num_family-1)]
    family_behaviors.append(yb_test[y_test == num_family - 1][0])
    run_explainer(x_train, y_train, x_test, y_test, dream_model, dis_exp_path, cert_loss_flag, concept_space, family_behaviors=family_behaviors, **kwargs)
    results = evaluate_explainers(x_train, y_train, x_test, y_test, dream_model, num_family, dis_exp_path, concept_space, cert_loss_flag, exp_name, **kwargs)
    return results


def run_explainer(x_train, y_train, x_test, y_test, dream_model, dis_exp_path, cert_loss_flag=True, concept_space=False, family_behaviors=None, debug=False, **kwargs):
    if os.path.exists(dis_exp_path):
        return 
    logging.info(f"Explanation masks will be saved to {dis_exp_path}")
    z_train, _ = dream_model.get_ae_outputs(x_train)
    z_centroids, closest_to_centroid_samples, mad_family, dis_family = prepare_training_centroids(x_train, z_train, y_train)
    
    x_drift = x_test[y_test == num_family - 1]
    num_drift_samples = len(x_drift)
    if cert_loss_flag:
        pred_labels = np.argmax(dream_model.predict(x_drift), axis=1)
        closest_families = pred_labels    
    else:    
        embeddings, _ = dream_model.get_ae_outputs(x_drift)
        closest_families = []
        for idx in range(len(x_drift)):
            dis_k = [np.linalg.norm(embeddings[idx] - z_centroids[i]) for i in range(num_family-1)]
            closest_families.append(np.argmin(dis_k))     

    dream_exp = []
    distance_lowerbound_list = []
    if debug: 
        family_ncms = get_train_ncm(x_train, y_train, dream_model, **kwargs)
    for idx, x_explain in tqdm(enumerate(x_drift), desc='Drift Explanation', total=num_drift_samples):
        closest_family = closest_families[idx]
        if debug and family_behaviors is not None:
            gt_drift = family_behaviors[-1]
            gt_expect = family_behaviors[closest_family]
            tqdm.write(f'Ground truth behaviors: Drift-{gt_drift}, Closest-{gt_expect}, Diff: {get_diff_idx(gt_drift, gt_expect)}')
        centroid_embedding = z_centroids[closest_family]
        closest_to_centroid_sample = closest_to_centroid_samples[closest_family]
        
        diff_idx = get_diff_idx(x_explain, closest_to_centroid_sample)
        distance_lowerbound = mad_family[closest_family] * kwargs.get('mad_threshold', 3.5) + np.median(dis_family[closest_family])
        distance_lowerbound_list.append(distance_lowerbound)

        mask = explain_instance(x_explain, dream_model, diff_idx, centroid_embedding, closest_to_centroid_sample, distance_lowerbound, kwargs.get('reg_lambda', 0.001), cert_loss=cert_loss_flag, concept_space=concept_space, closest_pred=dream_model.predict(closest_to_centroid_sample[None,:]), yb_centroid=family_behaviors[closest_family])
        mask = mask.astype(int)
        logging.debug(f'{mask.sum()} {f"concepts{np.where(mask==1)[0].tolist()}" if concept_space else "features"} are annotated as important.')
        dream_exp.append(mask)
        if debug:
            print('Mask > 0:', np.where(mask>0)[0])
            x_explain = np.array([x_explain]) if not concept_space \
                else dream_model.encoder.predict(np.array([x_explain]))
            closest_to_centroid_sample = np.array([closest_to_centroid_sample]) if not concept_space \
                else dream_model.encoder.predict(np.array([closest_to_centroid_sample]))
            tqdm.write(evaluate_drift_exp_credibility(family_ncms, x_explain, np.array([pred_labels[idx]]), np.array([mask]), closest_to_centroid_sample, dream_model, concept_space=concept_space))

    np.save(dis_exp_path, np.array(dream_exp))


def get_diff_idx(x0, x1):
    diff = x0 - x1
    diff_idx = np.where(diff != 0)[0]
    return diff_idx


def evaluate_explainers(x_train, y_train, x_test, y_test, dream_model, num_family, dis_exp_path, concept_space, cert_loss_flag, exp_name, **kwargs):
    z_train, _ = dream_model.get_ae_outputs(x_train)
    z_centroids, closest_to_centroid_samples, _, _ = prepare_training_centroids(x_train, z_train, y_train)
    
    x_drift = x_test[y_test == num_family - 1]
    num_drift_samples = len(x_drift)
    embeddings, reconstructions = dream_model.get_ae_outputs(x_drift)
    pred_labels = np.argmax(dream_model.predict(x_drift), axis=1)
    exp_baselines = np.array([closest_to_centroid_samples[i] for i in pred_labels])   
    
    # Cross Boundary Credibility based evaluation
    family_ncms = get_train_ncm(x_train, y_train, dream_model, **kwargs)
    dis_exp = np.load(dis_exp_path)
    num_masked_features = dis_exp.sum(-1)
    reference_pred = None
    if concept_space:
        x_drift = dream_model.encoder.predict(x_drift)
        reference_pred = dream_model.predict(exp_baselines)
        exp_baselines = dream_model.encoder.predict(exp_baselines)
    results = {}
    exp_cred = evaluate_drift_exp_credibility(family_ncms, x_drift, pred_labels, dis_exp, exp_baselines, dream_model, concept_space=concept_space)
    results[exp_name.capitalize()] = exp_cred
    logging.info(f'[Explanation Evaluation] Cross Boundary Cred. is {exp_cred}, annotated {num_masked_features.mean()} {"concepts" if concept_space else "features"}')

    diff = x_drift - exp_baselines
    diff_sample_ids, diff_pos_ids = np.where(diff != 0)
    ig_exp_path = dis_exp_path.replace('.npy', '_ig.npy')
    if os.path.exists(ig_exp_path):
        ig_attrs =  np.load(ig_exp_path)
    else:
        ig_attrs = dream_model.get_drift_ig_attr(x_drift, exp_baselines, concept_space=concept_space, reference_pred=reference_pred) 
        np.save(ig_exp_path, ig_attrs)
    ig_mask = np.array([top_k_mask(ig_attrs[idx], num_masked_features[idx], diff_pos_ids[diff_sample_ids==idx]) for idx in range(num_drift_samples)])
    ig_cred = evaluate_drift_exp_credibility(family_ncms, x_drift, pred_labels, ig_mask, exp_baselines, dream_model, concept_space=concept_space)
    results['Dri-IG'] = ig_cred
    logging.info(f'Drifting func + IG: {ig_cred}')
    
    if not concept_space:
        rec_mask = np.array([top_k_mask(np.abs(x_drift[idx] - reconstructions[idx]), num_masked_features[idx], diff_pos_ids[diff_sample_ids==idx]) for idx in range(num_drift_samples)])
        rec_cred = evaluate_drift_exp_credibility(family_ncms, x_drift, pred_labels, rec_mask, exp_baselines, dream_model)
        results['Recon.'] = rec_cred
        logging.info(f'Reconstruction Baseline: {rec_cred}')
    
    random_mask = generate_random_binary_masks(x_drift, num_masked_features)
    ran_cred = evaluate_drift_exp_credibility(family_ncms, x_drift, pred_labels, random_mask, exp_baselines, dream_model, concept_space=concept_space)
    results['Random'] = ran_cred
    logging.info(f'Random: {ran_cred}')

    # Centroid Distance based evaluation
    dis_result = {}
    if cert_loss_flag:
        closest_families = pred_labels
    else:
        closest_families = []
        for idx in range(len(x_drift)):
            dis_k = [np.linalg.norm(embeddings[idx] - z_centroids[i]) for i in range(num_family-1)]
            closest_families.append(np.argmin(dis_k))  
        exp_baselines = np.array([closest_to_centroid_samples[i] for i in closest_families])
        if concept_space: exp_baselines = dream_model.encoder.predict(exp_baselines)
    closest_centroids = np.array([z_centroids[i] for i in closest_families])
    ori_distances = np.linalg.norm(embeddings - closest_centroids, axis=1)
    dis_distances = get_perturbed_dis(dream_model, x_drift, dis_exp, exp_baselines, closest_centroids, single=False, concept_space=concept_space)
    dis_result[exp_name.capitalize()] = np.mean((ori_distances - dis_distances) / ori_distances)
    logging.info(f'[Explanation Evaluation] Centroid Distance: {np.mean(dis_distances):.3f} ± {np.std(dis_distances):.3f} (Original: {np.mean(ori_distances):.3f} ± {np.std(ori_distances):.3f})')

    ig_distances = get_perturbed_dis(dream_model, x_drift, ig_mask, exp_baselines, closest_centroids, single=False, concept_space=concept_space)
    dis_result['Dri-IG'] = np.mean((ori_distances - ig_distances) / ori_distances)
    logging.info(f'Drifting func + IG: {np.mean(ig_distances):.3f} ± {np.std(ig_distances):.3f}')

    if not concept_space:
        rec_distances = get_perturbed_dis(dream_model, x_drift, rec_mask, exp_baselines, closest_centroids, False)
        dis_result['Recon.'] = np.mean((ori_distances - rec_distances) / ori_distances)
        logging.info(f'Reconstruction Baseline: {np.mean(rec_distances):.3f} ± {np.std(rec_distances):.3f}')

    ran_distances = get_perturbed_dis(dream_model, x_drift, random_mask, exp_baselines, closest_centroids, False, concept_space=concept_space)
    dis_result['Random'] = np.mean((ori_distances - ran_distances) / ori_distances)
    logging.info(f'Random: {np.mean(ran_distances):.3f} ± {np.std(ran_distances):.3f}')

    return results, dis_result


def get_perturbed_dis(dream_model, x_explain, exp_mask, closest_to_centroid_sample, centroid_embedding, single=True, concept_space=False):
    perturbed_x = x_explain * (1 - exp_mask) + closest_to_centroid_sample * exp_mask
    perturbed_embedding = dream_model.get_ae_outputs(perturbed_x, single=single)[0] \
        if not concept_space else perturbed_x
    if single:
        perturbed_distance = np.linalg.norm(perturbed_embedding - centroid_embedding)
    else:
        perturbed_distance = np.linalg.norm(perturbed_embedding - centroid_embedding, axis=1)
    return perturbed_distance


def top_k_mask(arr, k, diff_idx):
    M_same = np.ones(arr.size, dtype=bool)
    M_same[diff_idx] = False
    arr[M_same] = np.min(arr)
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(arr)[::-1]    
    # Initialize a mask with zeros
    mask = np.zeros_like(arr, dtype=int)    
    # Set the top-k indices to 1
    mask[sorted_indices[:k]] = 1  
    return mask


def prepare_training_centroids(x_train, z_train, y_train):
    N, N_family, z_family = get_latent_data_for_each_family(z_train, y_train)
    centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
    # A list of length `num_families`; for each family `i`, `dis_family[i]` is the distance list between samples of this family and its centroids
    dis_family = get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family)
    mad_family = get_MAD_for_each_family(dis_family, N, N_family)
    closest_to_centroid_samples = []
    for family in range(N):
        dis_to_centroid_inds = np.array(dis_family[family]).argsort()
        x_train_family = x_train[np.where(y_train == family)[0]]
        closest_to_centroid_sample = x_train_family[dis_to_centroid_inds][0]        
        closest_to_centroid_samples.append(closest_to_centroid_sample)
    return centroids, closest_to_centroid_samples, mad_family, dis_family


# Custom logging handler that uses tqdm.write
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # Use tqdm's write function
        except Exception:
            self.handleError(record)


if __name__ == '__main__':
    data_name = args.data_name
    feature_name = args.feature_name
    model_name = 'dream'
    exp_name = 'cade'

    dataset_families = get_dataset_family(data_name)

    num_family = len(dataset_families)
    concept_space, reg_lambda = True, 0.01
    lambda_unc, lambda_con = 10., 1.
    # concept_space, reg_lambda = False, 0.001
    # lambda_unc, lambda_con = 10., None

    from datetime import datetime
    log_file = f"logs/explainer/{datetime.now().strftime('%m%d%H%M%S')}-{'[concept]' if concept_space else ''}{data_name}-{feature_name}-{model_name}-{exp_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            TqdmLoggingHandler()
        ] 
    )
    logging.info(f'concept_space: {concept_space}, reg_lambda: {reg_lambda}, lambda_unc: {lambda_unc}, lambda_con: {lambda_con}')

    cred_results = []
    dis_results = []
    for newfamily in range(num_family): # num_family-1,  
        logging.info(f"Hodling out on family {newfamily}-{dataset_families[newfamily]}")
        _result = main(data_name, feature_name, newfamily, num_family, model_name, exp_name, concept_space=concept_space, reg_lambda=reg_lambda, lambda_unc=lambda_unc, lambda_con=lambda_con)
        cred_results.append(_result[0])
        dis_results.append(_result[1])

    def add_mean_to_result(results):
        results = pd.DataFrame(results, index=dataset_families)
        means_df = pd.DataFrame([results.mean()], index=['Mean'])
        results = pd.concat([results, means_df]).T
        return results

    cred_results = add_mean_to_result(cred_results)
    logging.info(f'Cross Boundary P-values for Explainer [{exp_name}] on Model [{model_name}]: \n{cred_results}')

    dis_results = add_mean_to_result(dis_results)
    logging.info(f'Distance Reduction Rate for Explainer [{exp_name}] on Model [{model_name}]: \n{dis_results}')

