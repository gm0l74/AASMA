#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : drn.py
#
# @ start date          22 04 2020
# @ last update         15 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
from datetime import datetime
from PIL import Image
import numpy as np
import zmq, time

from aasma.agent.models.AgentModel import AgentModel
import aasma.agent.grabber as grabber

# Keras nn components
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

#---------------------------------
# Constants
#---------------------------------
IMG_SIZE = (600, 600)

#---------------------------------
# class DeepQNetwork
#---------------------------------
class DeepQNetwork:
    def __init__(self, actions, input_shape, load=None):
        # Hyper Parameters
        self.__gamma = 0.99
        self.__mini_batch_size = 32

        self.__model = None
        if load is None:
            # Build and compile the model
            self.build() ; self.__compile()
        else:
            # Load previously made model
            self.__load(load)

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
            32, 8, strides=(4, 4), padding='valid',
            activation='relu',
            input_shape=input_shape
        ))
        self.__model.add(Conv2D(
            64, 4, strides=(2, 2),
            padding='valid',
            activation='relu'
        ))
        self.__model.add(Conv2D(
            64, 3, strides=(1, 1),
            padding='valid',
            activation='relu'
        ))

        # Flatten and Fully connected
        self.__model.add(Flatten())
        self.__model.add(Dense(512, activation='relu'))
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
                x_train.append(datapoint['source'].astype(np.float64))

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

            cur_time = datetime.now().strftime("%H:%M:%S")
            print("[{}] loss {} | acc {}".format(
                cur_time, hist['loss'][0] , hist['acc'][0]
            ))

    def predict(self, data):
        if self.__model is None:
            raise ValueError('Model hasn\'t been built')
        else:
            if not isinstance(data, np.ndarray):
                raise ValueError('\'data\' must be np.array')

            data = data.astype(np.float64)
            self.model.predict(data, batch_size=1)[0]

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
    def __init__(self, actions):
        super(DeepReinforcementLearning, self).__init__(actions)

        # Parameters
        self.__epsilon_decrease_value = 0.99
        self.__min_epsilon = 0.1

        # Replay memory
        self.__experiences = []
        self.__training_count = 0

        # Create PolicyNet
        self.__PolicyNet = DeepQNetwork(
            self.actions, (4, 600, 600)
        )

        # Create ValueNet
        self.__ValueNet = DeepQNetwork(
            self.actions, (4, 600, 600)
        )

        # Reset value network
        self.__ValueNet.model.set_weights(self.__PolicyNet.model.get_weights())

    @property
    def experiences(self):
        return self.__experiences

    @property
    def training_count(self):
        return self.__training_count

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
        for i in xrange(self.__mini_batch_size):
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

#---------------------------------
# class AgentController
#---------------------------------
class AgentController(AgentModel):
    def __init__(self, actions):
        super(AgentController, self).__init__(actions)

        self.__agent = DeepQAgent(actions)

        # Parameters
        self.__max_episode_length = 200
        self.__update_frequency = 10
        self.__valueNet_update_freq = 30

        # Min n of transitions to store in the replay memory before training
        self.__replay_start_size = 3

    def perceive(self, snapshot):
        # Convert to gray-scale
        image = Image.fromarray(obs, 'RGB').convert('L').resize(IMG_SIZE)

        # Convert to a numpy array
        self.__last_snapshot = np.asarray(
            image.getdata(), dtype=np.uint8
        ).reshape(image.size[1], image.size[0])

        return self.__last_snapshot

    def make_action(self):
        # Always exploit
        # Don't use while training
        return np.argmax(self.__agent.predict(self.__last_snapshot))

    def __get_next_state(self, last, observation):
        # Next state is composed by:
        # - last 3 snapshots of the previous state
        # - new observation
        return np.append(last[1:], [observation], axis=0)

    def train(self):
        # Connect to env communicator
        ipc = zmq.Context().socket(zmq.REQ)

        ipc.setsockopt(zmq.LINGER, 0)
        ipc.setsockopt(zmq.AFFINITY, 1)
        ipc.setsockopt(zmq.RCVTIMEO, 3000) # 3 seconds timeout

        ipc.connect("tcp://localhost:5555")
        ipc.send(b"create")
        try:
            agent_id = ipc.recv().decode()
            if agent_id == 'nack':
                raise ValueError("Couldn't spawn agent")
        except:
            exit()

        # Training loop
        episode = 0 ; N_EPISODES = 1000
        while episode < N_EPISODES:
            # Observe reward and init first state
            observation = self.perceive(grabber.snapshot())

            # Init score system
            score = 0
            # Init state with the same observations
            state = np.array([ observation for _ in range(4) ])

            # Episode loop
            episode_step = 0
            while episode_step < self.__.max_episode_length:

                # Select an action using the agent
                action = agent.get_action(np.asarray([state]))

                # Send the selected action...
                query = "move,{},{}".format(agent_id, action)
                ipc.send(query.encode())

                #  ... and get the reward
                try:
                    reward = float(ipc.recv().decode())
                except:
                    # Communicator disbanded
                    exit()

                time.sleep(1/FPS)

                observation = self.perceive(grabber.snapshot())
                next_state = self.__get_next_state(state, observation)

                # Clip the reward
                clipped_reward = np.clip(reward, -1, 1)
                # Store transition in replay memory
                agent.add_experience(
                    np.asarray([state]),
                    action,
                    clipped_reward,
                    np.asarray([next_state])
                )

                score += reward

                # Train the agent
                do_update = episode_step % self.__update_frequency == 0
                exp_check = len(agent.experiences) >= self.__replay_start_size

                if do_update and exp_check:
                    agent.train()

                    # Every now and then, update ValueNet
                    if agent.training_count % self.__valueNet_update_freq == 0:
                        agent.reset_ValueNet()

                # Linear epsilon annealing
                if exp_check:
                    agent.update_epsilon()

                # Prepare for the next iteration
                state = next_state

                episode_step += 1
            episode += 1

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    AgentController().train()
