#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : DeepQ.py
#
# @ start date          21 05 2020
# @ last update         22 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import os, random
import numpy as np
from collections import deque

import aasma.agents.AgentAbstract as AgentAbstract
import aasma.utils as utils

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Layer
from tensorflow.keras.layers import Convolution2D, Dense, Flatten

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

#---------------------------------
# Constants
#---------------------------------
MAX_EPSILON = 1
MIN_EPSILON = 0.1
EXPLORATION_STEPS = 500000

LEARNING_RATE = 1e-5
DISCOUNT = 0.99

MAX_MEMORY_SIZE = 100000

# TODO review the next 175 lines of code

#---------------------------------
# Utilities
#---------------------------------
def dense_to_one_hot(data, depth=10):
    return (np.arange(depth) == np.array(data)[:, None]).astype(np.bool)

tf.compat.v1.disable_eager_execution()
#---------------------------------
# class DeepQ
#---------------------------------
class DeepQ(AgentAbstract.AgentAbstract):
    def __init__(self):
        super(DeepQ, self).__init__()

        self.batch = deque()
        self.q_out, self.policy = self.build_train_network()
        self.target = self.build_target_network()

        self.decay_step = 0
        self.learning_rate = LEARNING_RATE
        self.epsilon = MAX_EPSILON
        self.gamma = DISCOUNT

        self.episode = 0

        self.opt = Adam(lr=self.learning_rate)
        self.policy.compile(optimizer=self.opt, loss=[DeepQ.huber_loss])

    @property
    def mini_batch_size(self):
        return 32

    @property
    def alpha(self):
        return self.learning_rate

    def make_action(self, state):
        return self.policy.predict(state)

    def predict(self, state):
        q = self.q_out(state)
        return q, np.argmax(np.array(q).flatten())

    def feed_batch(self, data):
        # TODO
        if len(self.batch) >= MAX_MEMORY_SIZE:
            self.batch.popleft()
        self.batch.append(data)
        return self.batch

    def sample_batch(self):
        # TODO
        batch_q, batch_state, batch_mask, states_next, rewards, done =\
            map(lambda x: np.array(list(x)), zip(*random.sample(self.batch, 32)))
        batch_state = np.transpose(batch_state, axes=[0, 2, 3, 1])
        states_next = np.transpose(states_next, axes=[0, 2, 3, 1])
        batch_mask = dense_to_one_hot(batch_mask, len(utils.ACTIONS))
        q_next = self.target.predict(states_next)
        batch_q[batch_mask] = np.array(rewards) + self.gamma * np.array(done) * np.max(q_next, axis=1)
        return batch_q, batch_state, batch_mask

    def build_train_network():
        # TODO
        X = Input(shape=(80, 80, 4), dtype='float32')
        mask = Input(shape=(len(utils.ACTIONS),), dtype='float32')
        q_out, model = DeepQ.infer(X)
        q_ = Lambda(lambda x: K.reshape(K.sum(x * mask, axis=1), (-1, 1)), output_shape=(1,))(q_out)
        return K.function([X], [q_out]), Model([X, mask], q_)

    def build_target_network():
        # TODO
        X = Input(shape=(80, 80, 4), dtype='float32')
        Q, model = DeepQ.infer(X, trainable=False, init=initializers.zeros())
        return model

    @staticmethod
    def huber_loss(x, y):
        # TODO
        error = K.abs(x - y)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part, axis=-1)
        return loss

    @staticmethod
    def infer(X, trainable=True, init=initializers.TruncatedNormal(stddev=0.01)):
        # TODO
        init_w = init
        init_b = initializers.constant(0.)
        normed = Lambda(lambda x: x / 255., output_shape=K.int_shape(X)[1:])(X)
        h_conv1 = Convolution2D(32, (8, 8), strides=(4, 4),
                                kernel_initializer=init_w, use_bias=False, padding='same')(normed)
        h_ln1 = LayerNormalization(activation=K.relu)(h_conv1)
        h_conv2 = Convolution2D(64, (4, 4), strides=(2, 2),
                                kernel_initializer=init_w, use_bias=False, padding='same')(h_ln1)
        h_ln2 = LayerNormalization(activation=K.relu)(h_conv2)
        h_conv3 = Convolution2D(64, (3, 3), strides=(1, 1),
                                kernel_initializer=init_w, use_bias=False, padding='same')(h_ln2)
        h_ln3 = LayerNormalization(activation=K.relu)(h_conv3)
        h_flat = Flatten()(h_ln3)
        fc1 = Dense(512, use_bias=False, kernel_initializer=init_w)(h_flat)
        h_ln_fc1 = LayerNormalization(activation=K.relu)(fc1)
        z = Dense(len(utils.ACTIONS), kernel_initializer=init_w, use_bias=False, bias_initializer=init_b)(h_ln_fc1)
        model = Model(X, z)
        model.trainable = trainable
        return z, model

    def train(self, terminal_state):
        # TODO
        batch_q, batch_state, batch_mask = self.sample_batch()
        self.policy.fit([batch_state, batch_mask], np.sum(batch_mask * batch_q, axis=1), verbose=0)

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

        # Save for later use or for a possible training restore
        self.policy.save_weights(os.path.join(path, 'policy.h5'))
        self.target.save_weights(os.path.join(path, 'target.h5'))
        np.save(os.path.join(path, 'params'), [self.episode, self.epsilon])

    def restore_training(self, path='saved_models'):
        # TODO
        pass

#---------------------------------
# class LayerNormalization
#---------------------------------
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

    def get_config(self):
        config = {
            'eps': self.eps,
            'channels': self.channels,
            'activation': self.activation
        }

        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        dim = len(K.int_shape(inputs)) - 1
        mean = K.mean(inputs, axis=dim, keepdims=True)
        var = K.mean(K.square(inputs - mean), axis=dim, keepdims=True)
        outputs = (inputs - mean) / K.sqrt(var + self.eps)
        try:
            outputs = outputs * self.trainable_weights[0] + self.trainable_weights[1]
        except:
            pass
        if self.activation is None:
            return outputs
        else:
            return self.activation(outputs)

    def restore(self):
        # TODO
        pass
