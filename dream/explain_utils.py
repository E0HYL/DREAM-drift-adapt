#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   explain_utils.py
@Time    :   2024/01/11 14:23:17
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2024
@Desc    :   None
'''

# here put the import lib
import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def evaluate_drift_explanation(x_drift, exp_mask, nearest_sample, dream_model, reduction='mean', **kwargs):
    # the lower, the better
    concept_flag = kwargs.get('concept_space', False)
    perturbed_x = x_drift * (1 - exp_mask) + nearest_sample * exp_mask
    if concept_flag:
        perturbed_x = dream_model.decoder.predict(perturbed_x)
    drift_score = dream_model.get_drift_scores(perturbed_x, **kwargs)
    if reduction == 'none':
        return drift_score
    return drift_score.mean()


def evaluate_drift_exp_credibility(family_ncms, x_drift, predicted_labels, exp_mask, nearest_sample, dream_model, reduction='mean', **kwargs):
    # the higher, the better
    test_ncms = evaluate_drift_explanation(x_drift, exp_mask, nearest_sample, dream_model, reduction='none', **kwargs)
    family_ncms = np.array(family_ncms, dtype=object)
    reference_ncms = family_ncms[predicted_labels]
    cred_list = []
    for ncm, reference in zip(test_ncms, reference_ncms):
        greater_ncms = np.sum(reference >= ncm)
        credibility = greater_ncms / len(reference)
        cred_list.append(credibility)
    cred_list = np.array(cred_list)
    if reduction == 'none':
        return cred_list
    return cred_list.mean()


def get_train_ncm(x_train, y_train, dream_model, **kwargs):
    y_pred = dream_model.predict(x_train)
    predicted_labels = np.argmax(y_pred, axis=1)
    correct_filter = predicted_labels == y_train
    x_train = x_train[correct_filter]
    y_train = y_train[correct_filter]
    family_ncms = []
    for class_id in np.unique(y_train):
        family_x = x_train[y_train == class_id]
        family_ncms.append(dream_model.get_drift_scores(family_x, **kwargs))
    return family_ncms


def get_cross_dis_boundary_ratio(perturbed_dis, lowerbound_list):
    success_idx = np.where((perturbed_dis <= lowerbound_list) == True)[0]
    ratio = len(success_idx) / len(perturbed_dis)
    return ratio


def integrated_gradients(inputs, baseline, scoring_fn, steps=50, batch_size=32, **kwargs):
    """
    Compute integrated gradients for a batch of inputs for a given model with batching.

    :param inputs: Batch of input tensors or numpy arrays for which gradients are to be computed.
    :param model: TensorFlow/Keras model.
    :param baseline: Baseline input used in the integrated gradients calculation.
                     This should be broadcastable to the shape of inputs.
    :param steps: Number of steps for the integral approximation.
    :param batch_size: Size of the batch to process inputs to prevent memory overflow.
    :return: Integrated gradients for each input in the batch.
    """
    # Ensure inputs and baseline are tensors
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    # Generate the interpolation steps
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps)

    # Initialize integrated gradients
    integrated_gradients = np.zeros_like(inputs)

    # Iterate over each batch
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))
    reference_pred = kwargs.get('reference_pred', None)
    for batch_index in tqdm(range(num_batches), desc='IG Explaining (batch)'):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, inputs.shape[0])
        batch_inputs = inputs[start_index:end_index]
        batch_baseline = baseline[start_index:end_index] if baseline.shape == inputs.shape else baseline
        batch_reference_pred = None if reference_pred is None else reference_pred[start_index:end_index]

        # Initialize batch gradients
        batch_gradients = tf.zeros_like(batch_inputs)

        for alpha in alphas:
            with tf.GradientTape() as tape:
                # Compute interpolated inputs
                interpolated_input = batch_baseline + alpha * (batch_inputs - batch_baseline)
                tape.watch(interpolated_input)

                # Compute the loss / characteristic function
                score = scoring_fn(interpolated_input, reference_pred=batch_reference_pred)
                
            # Accumulate gradients
            gradients = tape.gradient(score, interpolated_input)
            batch_gradients += gradients / steps

        # Update integrated gradients for the batch
        integrated_gradients[start_index:end_index] = ((batch_inputs - batch_baseline) * batch_gradients).numpy()

    return integrated_gradients


def find_nearest_samples(X0, X1):
    """
    Find the nearest sample in X1 for each sample in X0 using a KD-Tree.

    Parameters:
    X0 (array-like): An array of samples.
    X1 (array-like): Another array of samples to compare against.

    Returns:
    array: An array of the nearest samples from X1 to each sample in X0.
    """
    # Create a KD-Tree with the samples from X1
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(X1)
    # Find the nearest neighbor in X1 for each sample in X0
    distances, indices = neighbors.kneighbors(X0)
    # Return the nearest samples
    return X1[indices.flatten()]


def generate_random_binary_masks(input_data, num_positive, dtype=int):
    """
    Generate a batch of random binary masks based on the shape of the input data.

    Parameters:
    input_data (numpy.ndarray): Input data for which masks are to be generated.
    num_positive (numpy.ndarray): The number of positive (1) values in each mask.

    Returns:
    numpy.ndarray: A batch of random binary masks.
    """
    batch_size, *input_shape = input_data.shape
    total_elements = np.prod(input_shape)
    
    masks = []
    num_idx = 0
    for _ in range(batch_size):
        flat_mask = np.zeros(total_elements, dtype=dtype)
        positive_indices = np.random.choice(total_elements, num_positive[num_idx], replace=False)
        flat_mask[positive_indices] = 1
        mask = flat_mask.reshape(input_shape)
        masks.append(mask)
        num_idx += 1

    return np.array(masks)
