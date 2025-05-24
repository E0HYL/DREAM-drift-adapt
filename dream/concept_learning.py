#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   concept_learning.py
@Time    :   2023/08/18 21:14:02
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
import tensorflow as tf
from dream.train_utils import ModelConfig, train_model, classification_loss, get_cls_pseudo_loss, get_cls_uncertainty, call_accepts_training, binary_crossentropy, categorical_focal_loss
from dream.train_sampler import get_default_sampler
from dream.explain_utils import integrated_gradients


def split_classifier(classifier, layer_idx):
    """
    Splits the classifier at the specified layer index.
    Parameters:
    - classifier: The original classifier model.
    - layer_idx: The index at which the classifier should be split. If 0, f0 will be the input and f1 will be the original classifier.
    Returns:
    - f0: The first part of the classifier up to the specified layer index.
    - f1: The second part of the classifier from the specified layer index to the end.
    """
    original_name = classifier.name
    try:
        in_shape = classifier.input_shape[1:]
    except AttributeError:
        in_shape = classifier.layers[0].input_shape[1:]
    input_layer = tf.keras.layers.Input(shape=in_shape, name='f0_input')
    if layer_idx == 0:
        # Special case where f0 is just the input and f1 is the entire classifier
        f0 = tf.keras.Model(inputs=input_layer, outputs=input_layer, name=f"{original_name}_f0")
        f1 = classifier
        f1._name = f"{original_name}_f1"
    else:
        # Split the classifier at the specified layer index
        x = input_layer
        for layer in classifier.layers[:layer_idx]:
            x = layer(x)
        f0 = tf.keras.Model(inputs=input_layer, outputs=x, name=f"{original_name}_f0", trainable=classifier.trainable)
        # Define f1
        f1_input = tf.keras.layers.Input(shape=classifier.layers[layer_idx].input_shape[1:], name='f1_input')
        x = f1_input
        for layer in classifier.layers[layer_idx:]:
            x = layer(x)
        f1 = tf.keras.Model(inputs=f1_input, outputs=x, name=f"{original_name}_f1", trainable=classifier.trainable)
        del classifier
    return f0, f1


def recover_classifier(f0, f1):
    new_model = tf.keras.Sequential()
    # Add layers from f0 to the new model
    for layer in f0.layers:
        new_model.add(layer) 
    # Add layers from f1 to the new model
    for layer in f1.layers:
        new_model.add(layer)
    return new_model


def safe_norm(x, epsilon=1e-10, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)


def reconstruction_loss(z, z_prime, reduction='auto'):
    return tf.keras.losses.MeanSquaredError(reduction=reduction)(z, z_prime)


def create_hierarchical_mask(labels, training, benign_in_weak_similar=True):
    if training:
        is_benign = tf.equal(labels, 0)
        is_malicious = tf.not_equal(labels, 0)

        malicious_different_family = tf.logical_and(
            tf.not_equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0)),
            tf.logical_and(
                tf.expand_dims(is_malicious, 1),
                tf.expand_dims(is_malicious, 0)
            )
        )
        same_family = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))

        if benign_in_weak_similar: # follow the original paper
            # P1: Same class (benign or malicious) but different families; weakly similar
            benign_match = tf.logical_and(
                tf.expand_dims(is_benign, 1),
                tf.expand_dims(is_benign, 0)
            )
            P1_mask = tf.logical_or(benign_match, malicious_different_family) 
            # P2: Same malware family; highly similar
            P2_mask = tf.logical_and(
                same_family,
                tf.logical_and(
                    tf.not_equal(tf.expand_dims(labels, 1), 0),
                    tf.not_equal(tf.expand_dims(labels, 0), 0)
                )
            )   
        else:
            P1_mask = malicious_different_family
            P2_mask = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
        # N: Benign vs. Malicious
        N_mask = tf.logical_or(
            tf.logical_and(tf.expand_dims(is_benign, 1), tf.logical_not(tf.expand_dims(is_benign, 0))),
            tf.logical_and(tf.logical_not(tf.expand_dims(is_benign, 1)), tf.expand_dims(is_benign, 0))
        )
    else:
        # equality of labels
        P1_mask = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
        P2_mask = tf.zeros_like(P1_mask)
        N_mask = ~P1_mask

    return P1_mask, P2_mask, N_mask


def concept_separation_loss(encoded_repr, labels, margin=10.0, hierarchical=True, training=True, **kwargs):
    distances = safe_norm(tf.expand_dims(encoded_repr, 1) - tf.expand_dims(encoded_repr, 0), axis=2)
    if hierarchical:
        benign_in_weak_similar = kwargs.get('benign_in_weak_similar', True)
        P1_mask, P2_mask, N_mask = create_hierarchical_mask(labels, training, benign_in_weak_similar)
        if not training:  # only preserve distances from the single test instance
            distances = distances[0]
            P1_mask, N_mask = P1_mask[0], N_mask[0]
        # weak similar; both BENIGN or both MALWARE but different family
        term_1 = tf.cond(
            tf.reduce_any(P1_mask),
            lambda: tf.reduce_mean(tf.maximum(0., tf.boolean_mask(distances, P1_mask) - margin)),
            lambda: 0.0
        )
        # strong similar: MALWARE with same family
        term_2 = tf.cond(
            tf.reduce_any(P2_mask),
            lambda: tf.reduce_mean(tf.boolean_mask(distances, P2_mask)),
            lambda: 0.0
        )
        term_3 = tf.cond(
            tf.reduce_any(N_mask),
            lambda: tf.reduce_mean(tf.maximum(0., 2 * margin - tf.boolean_mask(distances, N_mask))),
            lambda: 0.0
        )
        Lhc = term_1 + term_2 + term_3
    else:
        # CADE: Malware family separation
        positive_mask = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
        negative_mask = tf.math.logical_not(positive_mask)
        positive_loss = tf.cond(
            tf.reduce_any(positive_mask),
            lambda: tf.reduce_mean(tf.boolean_mask(distances, positive_mask)),
            lambda: 0.0
        )
        negative_loss = tf.cond(
            tf.reduce_any(negative_mask),
            lambda: tf.reduce_mean(tf.maximum(0., margin - tf.boolean_mask(distances, negative_mask))),
            lambda: 0.0
        )
        Lhc = positive_loss + negative_loss
    return Lhc


def concept_reliability_loss(original_preds, reconstructed_preds, reduction='auto', label_smoothing=0.1, focal=False, sample_weight=None, **kwargs):
    # difference between two probability distributions of the classifier (multi-class/binary)
    # label smoothing: makes the loss less sensitive to small differences between the distributions
    if focal:
        # reduction = 'sum'
        # handle class imbalance in malware detection: FL(p_t) = alpha * (1 − p_t)^gamma * CrossEntropy(y_true, y_pred)
        gamma = kwargs.get('gamma', 2.) # gamma > 0, forcing the model to focus on hard-to-classify examples
        return categorical_focal_loss(original_preds, reconstructed_preds, gamma=gamma, label_smoothing=label_smoothing, reduction=reduction, alpha=10. if reduction!='sum' else 1., sample_weight=sample_weight)
    else:
        return tf.keras.losses.CategoricalCrossentropy(reduction=reduction, label_smoothing=label_smoothing)(original_preds, reconstructed_preds, sample_weight=sample_weight)


def concept_presence_loss(concept_predictions, binary_labels, reduction='auto', from_logits=False, pos_weight=1.):
    # When concept_binary_labels are missing for a sample, they are represented with the special value -1.
    valid_labels_mask = tf.math.greater_equal(binary_labels, 0)  # Shape: [batch_size, n_c]
    safe_binary_labels = tf.where(valid_labels_mask, binary_labels, tf.zeros_like(binary_labels))

    if reduction == 'none' or reduction == 'auto':
        # Manually compute element-wise binary cross-entropy loss
        # pos_weight > 1 decreases the false negative count: labels * -log(sigmoid(logits)) * pos_weight + (1 - labels) * -log(1 - sigmoid(logits)); 
        elementwise_loss = binary_crossentropy(safe_binary_labels, concept_predictions, from_logits=from_logits, pos_weight=pos_weight)
        # Apply the mask to loss for invalid labels to 0
        masked_loss = tf.where(valid_labels_mask, elementwise_loss, tf.zeros_like(elementwise_loss))

        feature_axis = list(range(1, len(concept_predictions.shape))) # all axes except the batch axis (axis 0)
        # Compute the mean only over valid labels
        num_valid_labels = tf.reduce_sum(tf.cast(valid_labels_mask, dtype=elementwise_loss.dtype), axis=feature_axis)
        # avoid division by zero
        num_valid_labels = tf.where(tf.equal(num_valid_labels, 0), tf.ones_like(num_valid_labels), num_valid_labels)        
        batch_loss = tf.reduce_sum(masked_loss, axis=feature_axis) / num_valid_labels
        if reduction == 'none':
            return batch_loss
        return tf.reduce_mean(batch_loss) 
    else:
        return tf.keras.losses.BinaryCrossentropy(reduction=reduction, from_logits=from_logits)(safe_binary_labels, concept_predictions)


def mask_certain_concept(concept_probs, concept_binary_labels):
    valid_mask = tf.not_equal(concept_binary_labels, -1)
    masked_predictions = tf.boolean_mask(concept_probs, valid_mask)
    masked_labels = tf.boolean_mask(concept_binary_labels, valid_mask)
    return masked_predictions, masked_labels   


class SigmoidLayer(tf.keras.layers.Layer):
    def __init__(self, concept_cls, **kwargs):
        super(SigmoidLayer, self).__init__(**kwargs)
        self.concept_cls = concept_cls

    def call(self, inputs):
        if self.concept_cls is None:
            return inputs
        return tf.sigmoid(inputs)
    

class ContraBase(tf.keras.Model):
    def __init__(self):
        super(ContraBase, self).__init__()

    def multi2binary_malware_label_tensor(self, y):
        return tf.cast(tf.not_equal(y, 0), tf.int32)
        
    def classification_loss(self, preds, reduction):
        return classification_loss(self.cls_label, preds, reduction=reduction) 
    
    def concept_separation_loss(self, concept_scores, family_labels, hierarchical):
        return concept_separation_loss(concept_scores, family_labels, self.margin, hierarchical)
    
    def reconstruction_loss(z, z_prime, reduction):
        return reconstruction_loss(z, z_prime, reduction=reduction)


class DREAM(ContraBase):
    def __init__(self, classifier, encoder, decoder, num_concept=8, split_layer_idx=0, lambda_rec=0.1, lambda_sep=0.1, lambda_rel=1., lambda_pre=0.001, sep_margin=5., text_input=False, output_class=7, hierarchical=False, concept_cls=False, det_thr=None, eta=.1, **kwargs):
        super(DREAM, self).__init__()
        if classifier is not None:
            self.f0, self.f1 = split_classifier(classifier, split_layer_idx)
        else:
            self.f0 = kwargs.get('f0')
            self.f1 = kwargs.get('f1')
        self.encoder = encoder
        self.concept_cls = concept_cls
        if num_concept > 0 and concept_cls:
            self.concept_dense = tf.keras.layers.Dense(num_concept, activation='sigmoid')
        else:
            self.concept_dense = SigmoidLayer(concept_cls)
        self.concept_dense.build((None, num_concept))
        self.decoder = decoder

        self.num_concept = num_concept
        self.split_layer_idx = split_layer_idx
        self.lambda_rec = lambda_rec
        self.lambda_sep = lambda_sep
        self.lambda_rel = lambda_rel
        self.lambda_pre = lambda_pre
        self.margin = sep_margin
        self.text_input = text_input
        self.output_class = output_class
        self.hierarchical = hierarchical # for binary classification
        self.det_thr = det_thr
        self.eta = eta

        self.concept_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='concept_accuracy') 
        self.total_loss_metric = tf.keras.metrics.Mean(name='loss')
        self.classification_loss_metric = tf.keras.metrics.Mean(name='cls_loss')
        self.reconstruction_loss_metric = tf.keras.metrics.Mean(name='rec_loss')
        self.seperation_loss_metric = tf.keras.metrics.Mean(name='sep_loss')
        self.reliability_loss_metric = tf.keras.metrics.Mean(name='rel_loss')
        self.concept_presence_metric = tf.keras.metrics.Mean(name='pre_loss')      

    def call(self, inputs, training=False):
        z = self.f0(inputs, training=training)
        concept_embedding = self.encoder(z, training=training)
        concept_preds = self.concept_dense(concept_embedding[:, :self.num_concept])
        original_preds = self.f1(z, training=training)
        return original_preds
        
    def get_predictions(self, inputs, specify_output='all', batch_size=32):
        y_preds = []
        c_preds = []
        # Iterate over the data in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            # Pass inputs through the first part of the classifier to get z
            z = self.f0.predict(batch, batch_size=batch_size)
            # Encode the representation to get concept scores
            concept_embedding = self.encoder.predict(z, batch_size=batch_size)
            concept_preds = self.concept_dense(concept_embedding[:, :self.num_concept])
            c_preds.extend(concept_preds)
            if specify_output == 'exp':
                continue
            # Original predictions
            original_preds = self.f1.predict(z, batch_size=batch_size)
            y_preds.extend(original_preds)
        if specify_output == 'exp':
            return np.array(c_preds)
        return np.array(y_preds), np.array(c_preds)
    
    def get_ae_outputs(self, inputs, single=False):
        if single:
            inputs = inputs[np.newaxis, :]
        embedding = self.encoder.predict(self.f0.predict(inputs))
        reconstructed = self.decoder.predict(embedding)
        if single:
            return embedding[0], reconstructed[0]
        return embedding, reconstructed
    
    def f1_z_prime(self, z_prime, training=False):
        if self.split_layer_idx == 0 and self.text_input:
            for i, l in enumerate(self.f1.layers):
                if isinstance(l, tf.keras.layers.Embedding) or isinstance(l, tf.keras.layers.InputLayer):
                    if isinstance(l, tf.keras.layers.Embedding):
                        self.f1_embedding_idx = i
                    continue
                if call_accepts_training(l):
                    z_prime = l(z_prime, training=training)
                else:
                    z_prime = l(z_prime)
        else:
            z_prime = self.f1(z_prime, training=training)
        return z_prime
    
    def calculate_loss(self, inputs, family_labels, concept_binary_labels=None, training=True):
        # Pass inputs through the components
        z = self.f0(inputs, training=training)
        concept_scores = self.encoder(z, training=training)
        z_prime = self.decoder(concept_scores, training=training)
        # Original and reconstructed predictions
        original_preds = self.f1(z, training=training)
        reconstructed_preds = self.f1_z_prime(z_prime, training=training)
        reduction = 'auto' if training else 'none'
        # Compute the losses
        self.cls_label = self.multi2binary_malware_label_tensor(family_labels) if self.output_class == 2 else family_labels
        cls_loss = self.classification_loss(original_preds, reduction=reduction) 
        rec_loss = reconstruction_loss(self.f1.layers[self.f1_embedding_idx](z), z_prime, reduction=reduction)\
            if self.text_input else reconstruction_loss(z, z_prime, reduction=reduction)
        if self.output_class == 2: # and training
            if concept_binary_labels is not None:
                sample_weight = tf.math.sigmoid(tf.reduce_sum(concept_binary_labels, axis=-1))
            else:
                sample_weight = tf.math.sigmoid(tf.cast(~(family_labels==0), tf.float32))
        else:
            sample_weight = None
        rel_loss = concept_reliability_loss(original_preds, reconstructed_preds, reduction=reduction, label_smoothing=.0, sample_weight=sample_weight, focal=(self.output_class == 2))  # and training
        concept_probs = self.concept_dense(concept_scores[:, :self.num_concept])
        if concept_binary_labels is None:
            pre_loss = .0 
        else: 
            pre_loss = concept_presence_loss(concept_probs, concept_binary_labels, reduction=reduction, pos_weight=10. if (self.output_class == 2 and training) else 1., from_logits=self.concept_cls is None)
        if training:
            sep_loss = self.concept_separation_loss(concept_scores, family_labels, self.hierarchical)
            return original_preds, concept_probs, (cls_loss, rec_loss, sep_loss, rel_loss, pre_loss)
        else:
            return cls_loss, rec_loss, rel_loss, pre_loss

    def compile(self, **kwargs):
        lr = kwargs.get('lr', 4e-4)
        self.classifier_lr = kwargs.pop('classifier_lr', lr)
        self.detector_lr = kwargs.pop('detector_lr', lr)
        super(DREAM, self).compile(**kwargs)
        self.classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=self.classifier_lr)
        self.detector_optimizer = tf.keras.optimizers.Adam(learning_rate=self.detector_lr)

    def train_step(self, data):
        if len(data) == 2:
            inputs, (classifier_labels, concept_binary_labels) = data
        elif len(data) == 3:
            inputs, classifier_labels, concept_binary_labels = data

        cls_trainable = (self.f0.trainable or self.f1.trainable) if self.split_layer_idx else self.f1.trainable
        with tf.GradientTape(persistent=True) as tape:
            original_preds, concept_probs, (cls_loss, rec_loss, sep_loss, rel_loss, pre_loss) = self.calculate_loss(inputs, classifier_labels, concept_binary_labels)
            total_loss = self.lambda_rec * rec_loss + self.lambda_sep * sep_loss + self.lambda_rel * rel_loss + self.lambda_pre * pre_loss
            if cls_trainable:
                total_loss += cls_loss 

        if cls_trainable: 
            # Get the gradients
            classifier_trainable_vars = self.f0.trainable_variables + self.f1.trainable_variables
            cls_gradients = tape.gradient(total_loss, classifier_trainable_vars)
            # Apply the gradients
            self.classifier_optimizer.apply_gradients(zip(cls_gradients, classifier_trainable_vars))
        detector_trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables + self.concept_dense.trainable_variables
        det_gradients = tape.gradient(total_loss, detector_trainable_vars)
        if self.det_thr is not None:
            det_loss = self.lambda_rec * rec_loss + self.lambda_rel * rel_loss + self.lambda_pre * pre_loss
            # Check for concept stability
            if det_loss < self.det_thr and self.detector_optimizer.lr == self.detector_lr:
                new_learning_rate = self.detector_optimizer.lr * self.eta
                self.detector_optimizer.lr = new_learning_rate
                print(f"New learning rate for detector: {new_learning_rate}")
            # else:
            #     self.detector_optimizer.lr = self.detector_lr
        self.detector_optimizer.apply_gradients(zip(det_gradients, detector_trainable_vars))
        del tape
        
        if cls_trainable:
            self.compiled_metrics.update_state(self.cls_label, original_preds)
            self.classification_loss_metric.update_state(cls_loss)
        # Update custom metrics
        concept_probs, concept_binary_labels = mask_certain_concept(concept_probs, concept_binary_labels)
        self.concept_accuracy_metric.update_state(concept_binary_labels, concept_probs)
        # Update custom losses
        self.total_loss_metric.update_state(total_loss)
        self.reconstruction_loss_metric.update_state(rec_loss)
        self.seperation_loss_metric.update_state(sep_loss)
        self.reliability_loss_metric.update_state(rel_loss)
        self.concept_presence_metric.update_state(pre_loss)
        
        return {m.name: m.result() for m in self.metrics if m.count != 0.}
    
    def get_config(self):
        return {
            'num_concept': self.num_concept,
            'split_layer_idx': self.split_layer_idx,
            'lambda_rec': self.lambda_rec,
            'lambda_sep': self.lambda_sep,
            'lambda_rel': self.lambda_rel,
            'lambda_pre': self.lambda_pre,
            'sep_margin': self.margin,
            'text_input': self.text_input
            }

    def get_det_thr(self, inputs, classifier_labels, concept_binary_labels):
        _, rec_loss, rel_loss, pre_loss = self.calculate_loss(inputs, classifier_labels, concept_binary_labels, training=False)
        if len(rec_loss.shape) > 1:
            rec_loss = tf.reduce_mean(rec_loss, -1)
        return tf.reduce_mean(self.lambda_rec * rec_loss + self.lambda_rel * rel_loss + self.lambda_pre * pre_loss)
        
    def _get_drift_scores(self, inputs, method='tloss', **kwargs): 
        y_pred = self.predict(inputs)
        if method == 'basic':
            cls_uncertainty = get_cls_uncertainty(y_pred, entropy=(not self.hierarchical))
            return cls_uncertainty
        elif method == 'ploss':
            return get_cls_pseudo_loss(y_pred)
        elif method == 'tloss':
            pesudo_label = np.argmax(y_pred, axis=1)
            cls_loss, _, rel_loss, _ = self.calculate_loss(inputs, pesudo_label, None, False)
            rel_weight = kwargs.get('reliability_weight', self.lambda_rel)
            return cls_loss + rel_weight * rel_loss # rel_weight * rel_loss + cls_loss
    
    def get_drift_scores(self, x_data, batch_size=32, **kwargs):
        all_drift_scores = []
        # Iterate over the data in batches
        for i in range(0, len(x_data), batch_size):
            # Extract the current batch
            batch = x_data[i:i + batch_size]
            batch_drift_scores = self._get_drift_scores(batch, **kwargs)
            if isinstance(batch_drift_scores, tf.Tensor):
                batch_drift_scores = batch_drift_scores.numpy()
            all_drift_scores.extend(batch_drift_scores)
        return np.array(all_drift_scores) 
               
    def get_dream_scores(self, inputs, **kwargs):
        y_pred, y_concept_pred = self.get_predictions(inputs)
        pseudo_concept_label = np.where(y_concept_pred > .5, 1., 0.)
        pesudo_label = np.argmax(y_pred, axis=1)
        cls_loss, rec_loss, rel_loss, pre_loss = self.calculate_loss(inputs, pesudo_label, pseudo_concept_label, False)
        pre_loss = np.nanmean(1 - y_concept_pred, axis=1)
        rel_weight = kwargs.get('reliability_weight', self.lambda_rel)
        pre_weight = kwargs.get('concept_weight', self.lambda_pre)
        rec_weight = kwargs.get('reconstruction_weight', self.lambda_rec)
        cls_uncertainty = get_cls_uncertainty(y_pred, entropy=(not self.hierarchical))
        t0 = cls_uncertainty + pre_weight * pre_loss
        t1 = cls_uncertainty + rel_weight * rel_loss
        t2 = cls_uncertainty + rel_weight * rel_loss + rec_weight * rec_loss
        return np.array([cls_loss, rec_loss, rel_loss, pre_loss, t0, t1, t2])
    
    def get_neighbor_uncertainty(self, inputs, batch_size=10, scale=0.01):
        noise_batch = np.random.normal(loc=0.0, scale=scale, size=(batch_size-1,) + inputs.shape[1:])
        uncertainty_scores = []
        from tqdm import tqdm
        for _input in tqdm(inputs):
            noise_sample = _input + noise_batch
            noise_sample = np.vstack([_input, noise_sample])
            n_pred = self.f1(self.f0(noise_sample, training=False), training=False).numpy()
            pesudo_label = np.argmax(n_pred[0]) * np.ones(noise_sample.shape[:1])
            neighbor_uncertainty = classification_loss(pesudo_label, n_pred) #np.mean(get_cls_uncertainty(n_pred, entropy=(not self.hierarchical)))  
            # neighbor_uncertainty = np.mean(self.get_dream_scores(noise_sample))
            uncertainty_scores.append(neighbor_uncertainty)
        return np.array(uncertainty_scores)

    def scoring_fn(self, inputs, training=False, reduction='none', **kwargs):
        if self.concept_space: # explanation in concept space
            embedding = inputs
            y_pred = kwargs.get('reference_pred', None)
        else:
            input_mid = self.f0(inputs, training=training)
            y_pred = self.f1(input_mid, training=training)
            embedding = self.encoder(input_mid, training=training)
        reconstructed = self.decoder(embedding, training=training)
        rec_pred = self.f1_z_prime(reconstructed, training=training)
        score = -tf.reduce_sum(rec_pred * tf.math.log(rec_pred + 1e-10), axis=1)
        if reduction != 'none':
            score = tf.reduce_mean(score)
        if y_pred is not None:
            score += self.lambda_rel * concept_reliability_loss(y_pred, rec_pred, reduction=reduction)
        return score  
    
    def get_drift_ig_attr(self, x_drift, baseline, **kwargs):
        self.concept_space = kwargs.pop('concept_space', False)
        feature_attribution = integrated_gradients(x_drift, baseline, self.scoring_fn, **kwargs)
        return feature_attribution


def train_dream_model(X_train, y_train, y_concept_train, classifier, encoder, decoder, _model_path=None, save_bp=False, num_epochs=10, learning_rate=0.0004, cls_loss='sparse_categorical_crossentropy', cls_metric=['sparse_categorical_accuracy'], mon_metric=None, alternate_monitor='loss', batch_size=32, debug_mode=True, config_path=None, _continue=False, hierarchical=False, output_class=7, **kwargs):
    num_explicit_concept = y_concept_train.shape[-1]
    sampler = get_default_sampler(X_train, y_train, batch_size, y_concept_train, hierarchical)
    dataset = sampler.create_dataset()
    
    if config_path is not None: 
        mc = ModelConfig(config_path)
        config = mc.load_config()
        # kwargs = kwargs if num_epochs > 0 else {} # ignore if not training
        # Merge the JSON config dictionary with the kwargs dictionary, allowing any values in kwargs to overwrite those in config
        config = {**config, **kwargs, 'num_concept': num_explicit_concept}
    else:
        config = {**kwargs, 'num_concept': num_explicit_concept}
    dream_model = DREAM(classifier, encoder, decoder, hierarchical=hierarchical, output_class=output_class, **config)
    dream_config = dream_model.get_config()
    dream_model.build((None,) + X_train.shape[1:])
    
    model_path = _model_path.replace('.h5', f"_{config['lambda_rec']}_{config['lambda_sep']}_{config['lambda_rel']}_{config['lambda_pre']}_{config['sep_margin']}.h5") if save_bp else _model_path
    if num_epochs > 0:
        if config_path is not None: 
            mc.save_config(dream_config)
        if _continue and os.path.exists(_model_path):
            dream_model.load_weights(_model_path)
        logging.info(f"Training DREAM [{dream_config}] for {num_epochs} epochs. Model weights will be saved at '{model_path}'.")
        train_model(dataset, None, dream_model, batch_size=batch_size, num_epochs=num_epochs, model_path=model_path, loss=cls_loss, metrics=cls_metric, mon_metric=mon_metric, lr=learning_rate, debug=debug_mode, alternate_monitor=alternate_monitor, verbose_batch=False, steps_per_epoch=X_train.shape[0] // batch_size)
        if save_bp: dream_model.save_weights(_model_path)
    
    dream_model.load_weights(model_path)
    return dream_model
