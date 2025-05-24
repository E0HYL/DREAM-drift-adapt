#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_sampler.py
@Time    :   2023/10/25 19:26:13
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import tensorflow as tf
import numpy as np
from collections import Counter


class BaseSampler:
    def __init__(self, x_train, y_train, batch_size, yb_train=None, infinite=True):
        self.x_train = x_train
        self.labels = y_train
        self.yb_train = yb_train
        self.batch_size = batch_size
        self.infinite = infinite

    def generator(self):
        # yield indices
        raise NotImplementedError
    
    def fetch_data(self):
        for indices in self.generator():
            if self.yb_train is None:
                yield self.x_train[indices], self.labels[indices]
            else:
                yield self.x_train[indices], self.labels[indices], self.yb_train[indices]

    def create_dataset(self):
        input_shape = self.x_train.shape[1:]
        dataset_shape = (
            tf.TensorSpec(shape=(None, )+input_shape, dtype=tf.as_dtype(self.x_train.dtype)),
            tf.TensorSpec(shape=(None,), dtype=tf.as_dtype(self.labels.dtype))
        )
        if self.yb_train is not None:
            dataset_shape = (*dataset_shape, tf.TensorSpec(shape=(None, )+self.yb_train.shape[1:], dtype=tf.as_dtype(self.yb_train.dtype)))

        ds = tf.data.Dataset.from_generator(self.fetch_data, output_signature=dataset_shape)
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


"""
The first half is made of randomly chosen samples, and the second half is filled with samples that maintain the label distribution of the first half.
"""
class DistributedSampler(BaseSampler):
    def __init__(self, x_train, labels, batch_size, yb_train=None, upsample=None, infinite=True):
        super().__init__(x_train, labels, batch_size, yb_train=yb_train, infinite=infinite)
        assert (batch_size % 2 == 0), "batch_size must be an even number"
        self.upsample = upsample
        if upsample:
            self.all_indices = np.repeat(np.arange(len(labels)), upsample)
        else:
            self.all_indices = np.arange(len(labels))
        self.batch_half = batch_size // 2

    def generator(self):
        while True:
            np.random.shuffle(self.all_indices)
            for i in range(0, len(self.all_indices), self.batch_half):
                half_batch = self.all_indices[i: i + self.batch_half]
                label_counts = Counter(self.labels[half_batch])
                second_half_indices = []
                for label, count in label_counts.items():
                    label_indices = np.where(self.labels == label)[0]
                    sampled_label_indices = np.random.choice(label_indices, size=count, replace=True)
                    second_half_indices.extend(sampled_label_indices)
                indices = np.concatenate([half_batch, second_half_indices])
                yield indices

            if not self.infinite:
                break


"""
The first part of the batch is filled with randomly selected samples. 
The second part consists of a mix of similar (with the same label) and dissimilar (with different labels) samples. 
The proportion of similar samples is defined by similar_samples_ratio.
"""
class ContrastiveSampler(BaseSampler):
    def __init__(self, x_train, labels, batch_size, yb_train=None, similar_samples_ratio=0.25, infinite=True):
        super().__init__(x_train, labels, batch_size, yb_train=yb_train, infinite=infinite)
        if batch_size % 4 != 0:
            raise ValueError('batch_size should be a multiple of 4.')

        self.half_size = batch_size // 2
        self.similar_samples_ratio = similar_samples_ratio
        self.num_sim = int(self.half_size * similar_samples_ratio)
        self.num_dissim = self.half_size - self.num_sim
        self.num_classes = len(np.unique(labels))

        self.relabel_for_sampling()
        # Caching class-wise indices for efficiency
        self.index_cls = [np.where(self.labels == label)[0] for label in range(self.num_classes)]
        self.index_no_cls = [np.where(self.labels != label)[0] for label in range(self.num_classes)]

    def generator(self):
        while True:
            num_batches = len(self.labels) // self.batch_size
            random_idx = np.random.permutation(len(self.labels))

            for b in range(num_batches):
                # Get initial half of the batch
                indices = random_idx[b * self.half_size: (b + 1) * self.half_size].tolist()
                for m in range(self.half_size):
                    if m < self.num_sim:
                        # Similar sample
                        pair_idx = np.random.choice(self.index_cls[self.labels[indices[m]]])
                    else:
                        # Dissimilar sample
                        pair_idx = np.random.choice(self.index_no_cls[self.labels[indices[m]]])
                    indices.append(pair_idx)
                yield indices

            if not self.infinite:
                break

    def relabel_for_sampling(self):
        self.labels = relabel_for_unique(self.labels)


def relabel_for_unique(labels, hierarchical_check=False):
    unique_labels = np.unique(labels)
    label_has_data = {label: np.any(labels == label) for label in unique_labels}
    # Filter out labels with no data
    labels_with_data = [label for label, has_data in label_has_data.items() if has_data]
    # Create a new label mapping for continuous labels
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(labels_with_data)}
    if hierarchical_check:
        assert new_label_mapping[0] == 0, "Benign class should stay 0!"
    # Relabel the data
    new_labels = np.array([new_label_mapping[label] for label in labels])
    return new_labels    


def get_default_sampler(X_train, y_train, batch_size, y_concept_train, hierarchical=False):
    sampler = DistributedSampler(X_train, y_train, batch_size, y_concept_train) if hierarchical\
        else ContrastiveSampler(X_train, y_train, batch_size, y_concept_train)
    return sampler
    