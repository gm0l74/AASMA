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
    def __init__(self):
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        # Spawn the policy and target networks
        self.model = self.create_model()
        self.target_model = self.create_model()

        # Information on the nn structure
        self.model.summary()

    def create_model(self):
        # Build the model
        model = Sequential()

        # Convolution layers
        model.add(Conv2D(
            64, 5, strides=(2, 2),
            activation='relu',
            input_shape=(100, 100, 1)
        ))
        model.add(Conv2D(
            128, 5, strides=(2, 2),
            activation='relu'
        ))
        model.add(Conv2D(
            128, 5, strides=(2, 2),
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

    def make_action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return np.random.choice(len(utils.ACTIONS))

        state = state.reshape((1, *state.shape))
        return np.argmax(self.model.predict(state)[0])

    def add_memory(self, state, action, reward, new_state):
        self.memory.append([state, action, reward, new_state])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state = sample

            state = state.reshape((1, *state.shape))
            new_state = new_state.reshape((1, *new_state.shape))

            target = self.target_model.predict(state)

            Q_future = max(self.target_model.predict(new_state)[0])
            target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save(self):
        self.model.save_weights('policy.hdf5')
