#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_classifier.py
@Time    :   2023/07/26 14:36:29
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D, Dropout, Conv2D, MaxPooling2D, Flatten, Input, deserialize, serialize


class DAMDModel(Model):
    def __init__(self, no_tokens=218, no_labels=2, kernel_size_1=21, kernel_size_2=5, final_nonlinearity='softmax'):
        super(DAMDModel, self).__init__()
        # @param no_tokens: Number of tokens appearing in data
        # @param no_labels: Number of classes that can be predicted
        self.embedding = Embedding(no_tokens+1, output_dim=128)
        self.conv1 = Conv1D(filters=64, kernel_size=kernel_size_1, strides=2, padding='valid', activation='relu')
        self.conv2 = Conv1D(filters=64, kernel_size=kernel_size_2, strides=2, padding='valid', activation='relu')
        self.pool = GlobalMaxPooling1D()
        self.dense1 = Dense(64, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(no_labels, activation=final_nonlinearity)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


class MarkovCNN(Model):
    def __init__(self, input_shape=(10, 10, 1), num_classes=3):
        super(MarkovCNN, self).__init__(name='MarkovCNN')
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.maxpool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class DrebinMLP(Model):
    def __init__(self, dims=[100, 30, 8], activation='relu', dropout_rate=0.2, name='DrebinMLP', init_layer=None):
        super(DrebinMLP, self).__init__(name=name)
        self.dims = dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layers_list = [] if init_layer is None else [init_layer]

        # Build the layers
        for i in range(len(self.dims) - 2):
            self.layers_list.append(Dense(self.dims[i + 1], activation=self.activation, name='clf_%d' % i))
            if self.dropout_rate > 0:
                self.layers_list.append(Dropout(self.dropout_rate, seed=42))

        self.layers_list.append(Dense(self.dims[-1], activation='softmax', name='clf_%d' % (len(self.dims) - 2)))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

    
class MarkovMLP(DrebinMLP):
    def __init__(self, dims=[1000, 500, 8], activation='relu', dropout_rate=0.2):
        init_layer = Flatten(name='flat_input')
        super(MarkovMLP, self).__init__(dims, activation, dropout_rate, name='MarkovMLP', init_layer=init_layer)


def summarize_classifier(classifier, input_shape, silent=True):
    classifier.build((None,) + input_shape)
    classifier.call(Input(shape=input_shape))
    if not silent:
        return str(classifier.summary())


def modify_output(model, num_classes=None, freeze=False, out_name=None, name='new_model', activation='softmax', weights=None):
    oname = f'{model.layers[-1].name}_new' if out_name is None else out_name
    if freeze:
        for layer in model.layers:
            layer.trainable = False
    num_classes = num_classes if num_classes is not None else model.output_shape[-1] + 1
    new_output = Dense(num_classes, activation=activation, name=oname)(model.layers[-2].output)
    new_model = Model(inputs=model.layers[0].input, outputs=new_output, name=name)
    if weights is not None:
        new_model.layers[-1].set_weights(weights)
    return new_model


def clone_copy_model(initial_model, customized=True, **kwargs):
    # Keep initial_model unchanged
    # Clone the initial model's architecture and copy its weights
    model = initial_model.__class__(**kwargs) if customized else clone_model(initial_model)
    model.set_weights(initial_model.get_weights())
    return model


def copy_compiled_model(source_model, copy_weights=True):
    """Copy the compiled subclass of Model to Sequential"""
    target_model = Sequential()
    for source_layer in source_model.layers:
        # Serialize the source layer's configuration
        layer_config = serialize(source_layer)
        # Deserialize the layer configuration into a new layer
        new_layer = deserialize(layer_config)
        # Copy the weights from the source layer to the new layer
        if source_layer.weights and copy_weights:
            new_layer.build(source_layer.input_shape)
            new_layer.set_weights(source_layer.get_weights())
        # Add the new layer to the target model
        target_model.add(new_layer)
    target_model.build(source_model.layers[0].input_shape)
    try:
        target_model.compile(optimizer=source_model.optimizer, loss=source_model.loss, metrics=source_model.compiled_metrics._metrics)
    except AttributeError:
        pass
    return target_model


def get_expanded_weights(model):
    import numpy as np
    # Extract weights and biases from the last layer
    weights, biases = model.layers[-1].get_weights()
    num_input_features, num_classes = weights.shape
    # Using Xavier initialization for the new weights
    new_weights = np.random.randn(num_input_features, 1) * np.sqrt(2. / (num_input_features + 1))
    # Combine original weights with the new weights
    expanded_weights = np.concatenate([weights, new_weights], axis=1)
    new_bias = np.array([0.])
    expanded_biases = np.concatenate([biases, new_bias])
    return expanded_weights, expanded_biases
