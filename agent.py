#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : agent.py
#
# @ start date          16 05 2020
# @ last update         18 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import numpy as np
import random
from collections import deque

import utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#---------------------------------
# class DeepQAgent
#---------------------------------
class DeepQAgent:
    def __init__(self, weights_path=None):
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.005
        self.tau = .125

        # Spawn the policy and target networks
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        if weights_path is not None:
            self.load(weights_path)

        # Information on the nn structure
        self.model.summary()

    def create_model(self):
        # Build the model
        model = Sequential()

        # Convolution layers
        model.add(Conv2D(
            16, 5, strides=(2, 2),
            activation='relu',
            input_shape=(100, 100, 1)
        ))
        model.add(Conv2D(
            64, 5, strides=(2, 2),
            activation='relu'
        ))
        model.add(Conv2D(
            64, 5, strides=(2, 2),
            activation='relu'
        ))

        # Flatten and fully connected
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(len(utils.ACTIONS)))

        # Compile the model
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    def make_action(self, state, force=False):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        print("e ", self.epsilon)

        if (not force) and (np.random.random() < self.epsilon):
            return np.random.choice(len(utils.ACTIONS))

        state = state.reshape((1, *state.shape))
        return np.argmax(self.model.predict(state)[0])

    def add_memory(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = min(32, len(self.memory))

        STATE, ACTION, REWARD, NEXT_STATE, DONE = 0, 1, 2, 3, 4
        transitions = np.array(random.sample(self.memory, batch_size))

        states = np.array(transitions[:, STATE].tolist())
        Q = self.model.predict(states)
        unique_actions = np.unique(transitions[:, ACTION])
        for action in unique_actions:
            mask = transitions[:, ACTION] == action
            Q[mask, action] = transitions[mask, REWARD]
            ndone_mask = (~transitions[:, DONE] & mask).astype(bool)
            if np.any(ndone_mask == True):
                feed_state = np.array(transitions[ndone_mask, NEXT_STATE].tolist())
                target_prediction = self.target_model.predict(feed_state)
                Q[ndone_mask, action] += self.gamma*np.max(target_prediction, axis=1)
        hist = self.model.fit(states, Q, verbose=0)
        hist = hist.history
        print("Loss {}".format(hist['loss']))

    def target_train(self):
        weights = self.model.get_weights()
        # target_weights = self.target_model.get_weights()
        # for i in range(len(target_weights)):
        #     target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(weights)

    def save(self):
        self.model.save_weights('policy.hdf5')

    def load(self, weights_path):
        self.model.load_weights(weights_path)
