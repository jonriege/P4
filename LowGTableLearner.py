import numpy.random as npr
import math

# Fixed params
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# Training parameters
training_time_seconds = 30
mean_window = 50 # How many games to take average over to measure performance when training
training_iterations = 1000000  # Number of training iterations, we set it high and control with training time instead
tick_len = 1

# Hyper parameters for tuning
learning_rate = 0.15
lr_change = 0.0000
discount_factor = 0.95
epsilon = 0.00
epsilon_change = 0.00  # Epsilon change each tick

# Bucket granularity, separate sets for low and high gravity
low_tree_dist_bin_size = 0  # Can be set to 0, will then discard this variable
low_relative_dist_bin_size = 600
low_velocity_bucket_size = 600  # Can be set to binary, will then only look at sign

high_tree_dist_bin_size = 300  # Can be set to 0, will then discard this variable
high_relative_dist_bin_size = 20
high_velocity_bucket_size = 20  # Can be set to 'binary', will then only look at sign


class LowGTableLearner(object):

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 1  # Set gravity to 1 before it is calculated
        self.tick_number = 0  # Counter
        self.Q = dict()  # Q_table
        self.firstState = None
        self.epsilon = epsilon
        self.lr = learning_rate
        self.counter = 0

    def reset(self):
        # Reset parameters from last game
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 1
        self.tick_number = 0

        # Increment epoch counter
        self.counter += 1

        # Parameter decay
        self.decreaseEpsilon()
        self.decreaseLR()

    def action_callback(self, state):
        # Bucketize states
        new_state = self.simplifyState(state)

        # Initialize unvisited state to (0,0)
        if new_state not in self.Q:
            self.Q[new_state] = [0, 0]

        # Choose best policy from Q-table
        Q_state = self.Q[new_state]
        if Q_state[0] == Q_state[1]:
            new_action = 0
        elif Q_state[1] > Q_state[0]:
            new_action = 1
        else:
            new_action = 0

        # Epsilon greedy strategy, overwrite action with probability epsilon
        rand_Float = npr.uniform(0, 1)
        if rand_Float < self.epsilon:
            new_action = npr.choice([0, 1])

        # Don't jump if first tick, must calculate gravity
        if self.tick_number == 0:
            new_action = 0

        # C alculate gravity in second tick
        if self.tick_number == 1:
            self.gravity = int(abs(state['monkey']['vel'] - self.firstState['monkey']['vel']))

        # Update Q-table
        if self.tick_number != 0:
            self.Q[self.last_state][self.last_action] += learning_rate * \
                                                         ((self.last_reward + discount_factor * max(Q_state[0],
                                                                                                    Q_state[1])) -
                                                          self.Q[self.last_state][self.last_action])

        # Finished tick, update vars
        if self.tick_number == 0:
            self.firstState = state
        self.last_action = new_action
        self.last_state = new_state
        self.tick_number += 1
        return self.last_action

    def decreaseEpsilon(self):
        self.epsilon = self.epsilon * (1 - epsilon_change)
        return None

    def decreaseLR(self):
        self.lr = self.lr * (1 - lr_change)
        return None

    def getQ(self):
        return self.Q

    # Function to bucketize and discard unneeded vars
    def simplifyState(self, state):
        monkey_bottom = state['monkey']['bot']
        tree_bottom = state['tree']['bot']
        dist_difference = monkey_bottom - tree_bottom
        tree_dist = state['tree']['dist']
        monkey_vel = state['monkey']['vel']

        # The point of this is to bucketize differently based on calculated gravity
        if self.gravity == 4:
            relative_dist_bin_size = high_relative_dist_bin_size
            size = high_velocity_bucket_size
            tree_dist_bin_size = high_tree_dist_bin_size
            if high_tree_dist_bin_size == 0:
                tree_dist = 0
                tree_dist_bin_size = 1
        else:
            relative_dist_bin_size = low_relative_dist_bin_size
            size = low_velocity_bucket_size
            tree_dist_bin_size = low_tree_dist_bin_size
            if low_tree_dist_bin_size == 0:
                tree_dist = 0
                tree_dist_bin_size = 1

        return (self.discretizeVel(monkey_vel, size), int(tree_dist // tree_dist_bin_size),
                int(dist_difference // relative_dist_bin_size), self.gravity)

    # Function to bucketize velocity. Slightly different to implement 'binary'-opportunity.
    def discretizeVel(self, value, size):
        if size == 'Binary':
            # Find sign of velocity
            sign_vel = -1
            if value > 0:
                sign_vel = 1
            return sign_vel
        if value > 0:
            value = math.ceil(value / size)
        else:
            value = math.floor(value / size)
        return int(value)

    def reward_callback(self, reward):
        self.last_reward = reward