#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
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

import aasma.agents.AgentAbstract as AgentAbstract
import aasma.agents.LayerNormalization as LayerNormalization
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
MINI_BATCH_SIZE = 32

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

        optimizer = Adam(lr=self.learning_rate)

        # Custom loss function (Huber loss)
        # See https://medium.com/@gobiviswaml/huber-error-loss-functions-3f2ac015cd45
        def huber_loss(x, y):
            error = K.abs(x - y)
            quadratic = K.square(K.clip(error, 0.0, 1.0))
            linear = error - quadratic
            return K.mean(0.5 * quadratic + linear, axis=-1)

        self.policy.compile(optimizer=optimizer, loss=huber_loss)

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

    def sample_batch(self):
        # TODO
        batch_q, batch_state, batch_mask, states_next, rewards, done =\
            map(lambda x: np.array(list(x)), zip(*random.sample(self.memory, 32)))
        batch_state = np.transpose(batch_state, axes=[0, 2, 3, 1])
        states_next = np.transpose(states_next, axes=[0, 2, 3, 1])
        batch_mask = dense_to_one_hot(batch_mask, len(utils.ACTIONS))
        q_next = self.target.predict(states_next)
        batch_q = batch_q.reshape(32, 4)
        batch_q[batch_mask] = np.array(rewards) + self.gamma * np.array(done) * np.max(q_next, axis=1)
        return batch_q, batch_state, batch_mask

    def build_train_network(self):
        # TODO
        X = Input(shape=(*utils.IMG_SIZE, 4), dtype='float32')
        mask = Input(shape=(len(utils.ACTIONS),), dtype='float32')
        q_out, model = self.__build(X)
        q_ = Lambda(lambda x: K.reshape(K.sum(x * mask, axis=1), (-1, 1)), output_shape=(1,))(q_out)
        return K.function([X], [q_out]), Model([X, mask], q_)

    def build_target_network(self):
        X = Input(shape=(*utils.IMG_SIZE, 4), dtype='float32')
        _, model = self.__build(X, trainable=False, init=initializers.zeros())
        return model

    def __build(self, X, trainable=True, init=initializers.TruncatedNormal(stddev=0.01)):
        # TODO
        init_w = init
        init_b = initializers.constant(0.)
        normed = Lambda(lambda x: x / 255., output_shape=K.int_shape(X)[1:])(X)
        h_conv1 = Convolution2D(32, (8, 8), strides=(4, 4),
                                kernel_initializer=init_w, use_bias=False, padding='same')(normed)
        h_ln1 = LayerNormalization.LayerNormalization(activation=K.relu)(h_conv1)
        h_conv2 = Convolution2D(64, (4, 4), strides=(2, 2),
                                kernel_initializer=init_w, use_bias=False, padding='same')(h_ln1)
        h_ln2 = LayerNormalization.LayerNormalization(activation=K.relu)(h_conv2)
        h_conv3 = Convolution2D(64, (3, 3), strides=(1, 1),
                                kernel_initializer=init_w, use_bias=False, padding='same')(h_ln2)
        h_ln3 = LayerNormalization.LayerNormalization(activation=K.relu)(h_conv3)
        h_flat = Flatten()(h_ln3)
        fc1 = Dense(512, use_bias=False, kernel_initializer=init_w)(h_flat)
        h_ln_fc1 = LayerNormalization.LayerNormalization(activation=K.relu)(fc1)
        z = Dense(len(utils.ACTIONS), kernel_initializer=init_w, use_bias=False, bias_initializer=init_b)(h_ln_fc1)
        model = Model(X, z)
        model.trainable = trainable
        return z, model

    def learn(self):
        batch_q, batch_state, batch_mask = self.sample_batch()
        self.policy.fit(
            [batch_state, batch_mask],
            np.sum(batch_mask * batch_q, axis=1),
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

        # Save for later use or for a possible training restore
        self.policy.save_weights(os.path.join(path, 'policy.h5'))
        self.target.save_weights(os.path.join(path, 'target.h5'))
        np.save(os.path.join(path, 'params'), [self.episode, self.epsilon])

    def restore_training(self, path='saved_models'):
        episode, _, eps, __ = np.load(os.path.join(
            os.getcwd(), f'{path}/params.npy'
        ))

        self.episode = int(episode)
        self.epsilon = eps
