#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : DeepQ.py
#
# @ start date          21 05 2020
# @ last update         24 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import os, random
import numpy as np
from collections import deque

import aasma.utils as utils
import aasma.agents.AgentAbstract as AgentAbstract

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Layer
from tensorflow.keras.layers import Convolution2D, Dense, Flatten

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

tf.compat.v1.disable_eager_execution()

#---------------------------------
# Constants
#---------------------------------
MAX_EPSILON = 1
MIN_EPSILON = 0.1
EXPLORATION_STEPS = 500000

LEARNING_RATE = 1e-5
DISCOUNT = 0.99

MAX_MEMORY_SIZE = 100000
MINI_BATCH_SIZE = 32

#---------------------------------
# Utilities
#---------------------------------
def dense_to_one_hot(data):
    return (np.arange(len(utils.ACTIONS)) == \
        np.array(data)[:, None]).astype(np.bool)

#---------------------------------
# class LayerNormalization
#---------------------------------
# Inspired by
# https://github.com/IntoxicatedDING/DQN-Beat-Atari/blob/master/dqn.py
# and ported from tf1 to tensorflow.keras (tf2)
class LayerNormalization(Layer):
    def __init__(self, eps=1e-5, activation=None, **kwargs):
        self.eps = eps
        self.channels = None
        self.activation = activation

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        shape = [1] * (len(input_shape) - 1)
        shape.append(self.channels)
        self.add_weight('gamma', shape, dtype='float32', initializer='ones')
        self.add_weight('beta', shape, dtype='float32', initializer='zeros')

        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dim = len(K.int_shape(inputs)) - 1
        mean = K.mean(inputs, axis=dim, keepdims=True)
        var = K.mean(K.square(inputs - mean), axis=dim, keepdims=True)
        outputs = (inputs - mean) / K.sqrt(var + self.eps)

        try:
            outputs = outputs * self.trainable_weights[0] + \
                self.trainable_weights[1]
        except:
            pass

        if self.activation is None:
            return outputs
        else:
            return self.activation(outputs)

    def get_config(self):
        config = {
            'eps': self.eps,
            'channels': self.channels,
            'activation': self.activation
        }

        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#---------------------------------
# class DeepQ
#---------------------------------
class DeepQ(AgentAbstract.AgentAbstract):
    def __init__(self, load=None):
        super(DeepQ, self).__init__()

        # Memory
        self.memory = deque()

        # Neural Networks
        self.q_out, self.policy = self.build_train_network()
        self.target = self.build_target_network()

        # Hyper parameters
        self.learning_rate = LEARNING_RATE
        self.epsilon = MAX_EPSILON
        self.gamma = DISCOUNT

        self.episode = 0

        # Custom loss function (Huber loss)
        # See https://medium.com/@gobiviswaml/huber-error-loss-functions-3f2ac015cd45
        def huber_loss(x, y):
            error = K.abs(x - y)
            quadratic = K.square(K.clip(error, 0.0, 1.0))
            linear = error - quadratic
            return K.mean(0.5 * quadratic + linear, axis=-1)

        self.policy.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=huber_loss
        )

        if load is not None:
            pwd = os.getcwd()

            self.policy.load_weights(os.path.join(
                pwd, f'{load}/policy.h5'
            ))
            self.target.load_weights(os.path.join(
                pwd, f'{load}/target.h5'
            ))

    @property
    def mini_batch_size(self):
        return MINI_BATCH_SIZE

    @property
    def alpha(self):
        return self.learning_rate

    def make_action(self, state):
        _, action = self.policy.predict(state)
        return action

    def predict(self, state):
        q = self.q_out(state)
        return q, np.argmax(np.array(q).flatten())

    def add_experience(self, experience):
        if len(self.memory) >= MAX_MEMORY_SIZE:
            # Free oldest experience
            self.memory.popleft()

        self.memory.append(experience)
        return self.memory

    def create_mini_batch(self):
        q_values, states, mask, next_states, rewards, done = map(
            lambda x: np.array(list(x)),
            zip(*random.sample(self.memory, MINI_BATCH_SIZE))
        )

        # Pre-process data structures before
        # returning them to be trained on
        q_values = q_values.reshape(MINI_BATCH_SIZE, 4)
        states = np.transpose(states, axes=[0, 2, 3, 1])
        next_states = np.transpose(next_states, axes=[0, 2, 3, 1])

        # One hot encode labels
        # See https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
        mask = dense_to_one_hot(mask)
        q_next = self.target.predict(next_states)

        # Update the q-values of the mini-batch
        q_values[mask] = np.array(rewards) + \
            self.gamma * np.array(done) * np.max(q_next, axis=1)
        return q_values, states, mask

    def build_train_network(self):
        x = Input(shape=(*utils.IMG_SIZE, 4), dtype='float32')
        mask = Input(shape=(len(utils.ACTIONS),), dtype='float32')

        q_out, model = self.__build(x)
        q_ = Lambda(
            lambda x: K.reshape(K.sum(x * mask, axis=1), (-1, 1)),
            output_shape=(1,)
        )(q_out)
        return K.function([x], [q_out]), Model([x, mask], q_)

    def build_target_network(self):
        x = Input(shape=(*utils.IMG_SIZE, 4), dtype='float32')
        _, network = self.__build(
            x, trainable=False,
            init=initializers.zeros()
        )
        return network

    def __build(
        self, _input, trainable=True,
        init=initializers.TruncatedNormal(stddev=0.01)
    ):
        # Kernel initializers and bias initialers
        init_weights = init
        init_biases = initializers.constant(0.)

        # Neural net architecture inspired by...
        # https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        # ... and...
        # https://github.com/IntoxicatedDING/DQN-Beat-Atari/blob/master/dqn.py

        # Normalize the input colors
        # Keep them in [0, 1]
        output = Lambda(
            lambda x: x / 255., output_shape=K.int_shape(_input)[1:]
        )(_input)
        output = Convolution2D(
            32, (8, 8), strides=(4, 4),
            padding='same',
            kernel_initializer=init_weights,
            use_bias=False
        )(output)
        output = LayerNormalization(activation=K.relu)(output)
        output = Convolution2D(
            64, (4, 4), strides=(2, 2),
            padding='same',
            kernel_initializer=init_weights,
            use_bias=False
        )(output)
        output = LayerNormalization(activation=K.relu)(output)
        output = Convolution2D(
            64, (3, 3), strides=(1, 1),
            padding='same',
            kernel_initializer=init_weights,
            use_bias=False
        )(output)
        output = LayerNormalization(activation=K.relu)(output)
        output = Flatten()(output)

        output = Dense(
            512, use_bias=False,
            kernel_initializer=init_weights
        )(output)
        output = LayerNormalization(
            activation=K.relu
        )(output)

        output = Dense(
            len(utils.ACTIONS),
            kernel_initializer=init_weights,
            use_bias=False,
            bias_initializer=init_biases
        )(output)

        model = Model(_input, output)
        model.trainable = trainable
        return output, model

    def learn(self):
        # Obtain a mini-batch with size MINI_BATCH_SIZE
        q, state, mask = self.create_mini_batch()

        # Use that mini-batch to train the network
        self.policy.fit(
            [state, mask],
            np.sum(batch_mask * q, axis=1),
            verbose=0
        )

    def update_epsilon(self):
        self.epsilon = np.maximum(
            MIN_EPSILON,
            self.epsilon - (MAX_EPSILON - MIN_EPSILON) / EXPLORATION_STEPS
        )

    def update_target_network(self):
        self.target.set_weights(self.policy.get_weights())

    def save_weights(self, path='saved_models'):
        if not os.path.exists(path):
            os.mkdir(path)

        # Save for later use or for possible training restore
        self.policy.save_weights(os.path.join(path, 'policy.h5'))
        self.target.save_weights(os.path.join(path, 'target.h5'))
        np.save(os.path.join(path, 'params'), [self.episode, self.epsilon])

    def restore_training(self, path='saved_models'):
        # Support two versions of models
        # One version has four parameters and
        # the other has two.
        # In either case, only two parameters are required
        unpack = np.load(os.path.join(
            os.getcwd(), f'{path}/params.npy'
        ))
        n_parameters = len(unpack)

        if n_parameters == 2:
            # Second version
            episode, eps = unpack
        elif n_parameters == 4:
            # First version
            episode, _, eps, __ = unpack
        else:
            raise ValueError('Arguments unpacking error')

        self.episode = int(episode)
        self.epsilon = eps
