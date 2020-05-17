#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : agent.py
#
# @ start date          16 05 2020
# @ last update         17 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
from datetime import datetime
from PIL import Image
import numpy as np
import time
from random import random, randint, randrange

# Keras nn components
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense

#---------------------------------
# Constants
#---------------------------------
IMG_SIZE = (600, 600)

#---------------------------------
# class DeepQNetwork
#---------------------------------
class DeepQNetwork:
    def __init__(self, actions, input_shape, load=None):
        self.actions = actions
        self.__input_shape = input_shape

        # Hyper Parameters
        self.__gamma = 0.99
        self.__mini_batch_size = 32

        self.__model = None
        if load is None:
            # Build and compile the model
            self.__build() ; self.__compile()
            self.summarize()
        else:
            # Load previously made model
            self.load(load)

    @property
    def model(self):
        return self.__model

    def summarize(self):
        print(" => Summary")
        self.__model.summary()

    def __build(self):
        n_actions = len(self.actions)

        # Build the neural net
        print(" => Building model")
        self.__model = Sequential()

        # Convolution layers (stack three of them)
        self.__model.add(Conv2D(
            16, 4, strides=(4, 4), padding='valid',
            activation='relu',
            input_shape=self.__input_shape
        ))
        self.__model.add(Conv2D(
            32, 2, strides=(2, 2),
            padding='valid',
            activation='relu'
        ))
        # self.__model.add(Conv2D(
        #     64, 3, strides=(1, 1),
        #     padding='valid',
        #     activation='relu'
        # ))
        #self.__model.add(MaxPool2D((2, 2)))

        # Flatten and Fully connected
        self.__model.add(Flatten())
        self.__model.add(Dense(128, activation='relu'))
        self.__model.add(Dense(n_actions))

    def __compile(self):
        if self.__model is None:
            raise ValueError('Model hasn\'t been built')
        else:
            print(" => Compiling model")
            self.__model.compile(
                loss='mse',
                optimizer='rmsprop', metrics=['accuracy']
            )

    def train(self, batch, ValueNet):
        if self.__model is None:
            raise ValueError('Model hasn\'t been built')
        else:
            x_train, y_train = [], []

            # Generate inputs and targets
            for datapoint in batch:
                x_train.append(
                    datapoint['source'].astype(np.float64).reshape(1, 600, 600, 4)
                )

                # Obtain the q value of the state
                next_state = datapoint['destination'].astype(np.float64)
                next_q_value = np.max(ValueNet.predict(next_state))

                y = list(self.predict(datapoint['source']))

                # Calculate rewards
                y[datapoint['action']] = datapoint['reward'] + \
                    self.__gamma * next_q_value

                y_train.append(y)

            # Convert training lists to numpy arrays
            x_train = np.asarray(x_train).squeeze()
            y_train = np.asarray(y_train).squeeze()

            hist = self.__model.fit(
                x_train, y_train,
                batch_size=self.__mini_batch_size,
                epochs=1, verbose=0
            )

            hist = hist.history
            cur_time = datetime.now().strftime("%H:%M:%S")
            print("[{}] loss {} | acc {}".format(
                cur_time, hist['loss'][0] , hist['accuracy'][0]
            ))

    def predict(self, data):
        if self.__model is None:
            raise ValueError('Model hasn\'t been built')
        else:
            if not isinstance(data, np.ndarray):
                raise ValueError('\'data\' must be np.array')

            data = data.astype(np.float64).reshape(1, 600, 600, 4)
            return self.__model.predict(data)[0]

    def load(self, filename):
        print(" => Loading model...")
        print(" => File: {}".format(filename))
        try:
            self.__model = load_model(filename)
        except:
            raise ValueError("FileIO exception")

    def load_weights(self, filename):
        print(" => Loading weights...")
        print(" => File: {}".format(filename))
        try:
            self.__model.load_weights(filename)
        except:
            raise ValueError("FileIO exception")

    def save(self, filename):
        print(" => Saving model...")
        print(" => File: {}".format(filename))
        self.__model.save(filename)

    def save_weights(self, filename):
        print(" => Saving weights...")
        print(" => File: {}".format(filename))
        self.__model.save_weights(filename)

#---------------------------------
# class DeepQAgent
#---------------------------------
class DeepQAgent:
    def __init__(self, actions, load=[None, None]):
        self.actions = actions

        # Parameters
        self.__epsilon = 1
        self.__epsilon_decrease_value = 0.001
        self.__min_epsilon = 0.1
        self.__replay_mem_size = 1024
        self.__mini_batch_size = 32

        # Replay memory
        self.__experiences = []
        self.__training_count = 0

        # Create PolicyNet
        self.__PolicyNet = DeepQNetwork(
            self.actions, (600, 600, 4), load=load[0]
        )

        # Create ValueNet
        self.__ValueNet = DeepQNetwork(
            self.actions, (600, 600, 4), load=load[1]
        )

        # Reset value network
        self.__ValueNet.model.set_weights(self.__PolicyNet.model.get_weights())

    @property
    def experiences(self):
        return self.__experiences

    @property
    def training_count(self):
        return self.__training_count

    def predict(self, snapshot):
        return np.argmax(self.__PolicyNet.predict(snapshot))

    def get_action(self, snapshot, pick_random=False):
        is_random = random() < self.__epsilon
        if pick_random or is_random:
            return randint(0, len(self.actions) - 1)
        else:
            return np.argmax(self.__PolicyNet.predict(snapshot))

    def compute_q_max(self, snapshot):
        # Given a snapshot, return the highest q-value
        q_values = self.__PolicyNet.predict(snapshot)

        # Error handle multiple actions with the same highest q
        q_values = np.argwhere(q_values == np.max(q_values))
        return random.choice(q_values)

    def get_random_snapshot(self):
        random_snap_id = randrange(0, len(self.__experiences))
        return self.__experiences[random_snap_id]['source']

    def add_experience(self, source, action, reward, destination):
        # Free older snapshots in replay memory
        if len(self.__experiences) >= self.__replay_mem_size:
            self.__experiences.pop(0)

        self.__experiences.append({
            'source': source,
            'action': action,
            'reward': reward,
            'destination': destination
        })

    def sample_batch(self):
        batch = []
        for i in range(self.__mini_batch_size):
            batch.append(self.__experiences[
                randrange(0, len(self.__experiences))
            ])

        return np.asarray(batch)

    def train(self):
        self.__training_count += 1
        self.__PolicyNet.train(self.sample_batch(), self.__ValueNet)

    def update_epsilon(self):
        # Gradually decrease the probability of picking a random action
        if self.__epsilon - self.__epsilon_decrease_value > self.__min_epsilon:
            self.__epsilon -= self.__epsilon_decrease_value
        else:
            self.__epsilon = self.__min_epsilon

    def reset_ValueNet(self):
        self.__ValueNet.model.set_weights(self.__PolicyNet.model.get_weights())

    def image_to_numpy(self, snapshot):
        snapshot = Image.fromarray(
            snapshot, 'RGB'
        ).convert('L').resize(IMG_SIZE)
        self.snapshot = np.asarray(
            snapshot.getdata(), dtype=np.uint8
        ).reshape(image.size[1], image.size[0])

    def save_progress(self):
        self.__ValueNet.save('value_net.h5')
        self.__PolicyNet.save('policy_net.h5')
