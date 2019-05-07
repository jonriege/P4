import numpy.random as npr
import numpy as np
from tensorflow import keras
import random

ACTIONS = [0, 1]

lr = 0.01          # Learning rate
lr_decay = 0.0001  # Learning rate decay
df = 0.7           # Discount factor
e = 0.1            # Greedy factor
e_decay = 0.001    # Greedy factor decay
mem_size = 100     # Memory size for experience replay
batch_size = 15    # Batch size when updating NN


class ReplayNNLearner(object):

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None
        self.e = e
        self.memory = []

        # Compile four NN models, one per action per gravity state.
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(batch_input_shape=(1, 6)))
        model.add(keras.layers.Dense(12, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear', kernel_initializer='zeros',
                                        bias_initializer='zeros'))
        sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay)
        self.models = [
            [keras.models.clone_model(model), keras.models.clone_model(model)],
            [keras.models.clone_model(model), keras.models.clone_model(model)]
        ]

        for i in range(2):
            for j in range(2):
                self.models[i][j].compile(loss='mse', optimizer=sgd)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

    # Q-function, predict present value given state and action
    def Q(self, state, action):
        s = self.normalized_state(state)[np.newaxis, :]
        model = self.models[self.gravity][action]
        return model.predict(s)[0][0]

    # Infer the gravity in the current epoch
    def infer_gravity(self, new_state):
        gv = self.last_state['monkey']['vel'] - new_state['monkey']['vel']
        self.gravity = int((gv - 1) / 3)

    # Return a normalized array containing the current state
    def normalized_state(self, state):
        return np.array([
            state['monkey']['bot'] / 400.0,
            (state['monkey']['vel'] + 40.0) / 60.0,
            state['tree']['dist'] / 600.0,
            (state['monkey']['bot'] - state['tree']['bot']) / 200.0,
            (state['tree']['top'] - state['monkey']['top']) / 200.0,
            self.gravity
        ])

    # Update the memory buffer with the new state
    def update_memory(self, new_state):
        last_s = self.normalized_state(self.last_state)[np.newaxis, :]
        new_s = self.normalized_state(new_state)[np.newaxis, :]
        self.memory.append(np.array([last_s, self.last_action, self.last_reward, new_s]))
        self.memory = self.memory[-mem_size:]

    def action_callback(self, new_state):

        if self.gravity is None and self.last_state is not None:
            self.infer_gravity(new_state)

        # Choose new action
        if self.last_state is None:
            new_action = 0
        elif npr.random() < self.e:
            new_action = npr.choice((0, 1), p=(0.9, 0.1))
        else:
            values = [self.Q(new_state, a) for a in ACTIONS]
            new_action = np.argmax(values)

        # Update Q-function
        if self.last_state is not None:
            self.update_memory(new_state)
            k = min(batch_size, len(self.memory))
            train_batch = random.choices(self.memory, k=k)

            for s, a, r, new_s in train_batch:
                gv = int(s[0][5])
                target = r + df * max(self.models[gv][new_a].predict(new_s) for new_a in ACTIONS)
                self.models[gv][a].train_on_batch(s, [target])

        # Update params
        self.last_action = new_action
        self.last_state = new_state
        if self.last_state is None:
            self.e -= e_decay

        return self.last_action

    def reward_callback(self, reward):
        self.last_reward = reward


class ReplayNNTrained(object):

    def __init__(self):
        self.gravity = None
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # Compile four NN models, one per action per gravity state.
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(batch_input_shape=(1, 6)))
        model.add(keras.layers.Dense(12, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear', kernel_initializer='zeros',
                                        bias_initializer='zeros'))
        sgd = keras.optimizers.SGD(lr=lr, decay=lr_decay)
        self.models = [
            [keras.models.clone_model(model), keras.models.clone_model(model)],
            [keras.models.clone_model(model), keras.models.clone_model(model)]
        ]

        for i in range(2):
            for j in range(2):
                model = self.models[i][j]
                model.load_weights(f"replay_nn_{i}{j}.h5")
                model.compile(loss='mse', optimizer=sgd)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

    # Q-function, predict present value given state and action
    def Q(self, state, action):
        s = self.normalized_state(state)[np.newaxis, :]
        model = self.models[self.gravity][action]
        return model.predict(s)[0][0]

    # Return a normalized array containing the current state
    def normalized_state(self, state):
        return np.array([
            state['monkey']['bot'] / 400.0,
            (state['monkey']['vel'] + 40.0) / 60.0,
            state['tree']['dist'] / 600.0,
            (state['monkey']['bot'] - state['tree']['bot']) / 200.0,
            (state['tree']['top'] - state['monkey']['top']) / 200.0,
            self.gravity
        ])

    # Infer the gravity in the current epoch
    def infer_gravity(self, new_state):
        gv = self.last_state['monkey']['vel'] - new_state['monkey']['vel']
        self.gravity = int((gv - 1) / 3)

    def action_callback(self, new_state):

        if self.gravity is None and self.last_state is not None:
            self.infer_gravity(new_state)

        # Choose new action
        if self.last_state is None:
            new_action = 0
        else:
            values = [self.Q(new_state, a) for a in ACTIONS]
            new_action = np.argmax(values)

        # Update params
        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        self.last_reward = reward
