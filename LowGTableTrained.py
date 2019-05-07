import math
import pickle

# Fixed params
SCREEN_WIDTH  = 600
SCREEN_HEIGHT = 400

# Testing parameters
training_time_seconds = 180
mean_window = 300  # How many games to take average over to measure performance
training_iterations = 1000000  # Number of training iterations, we set it high and control with training time instead
tick_len = 10


def load_obj(name ):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)


# Load Q_table and set bucket_size
Q_table = load_obj("Q-table_short")
high_params = Q_table['high_params']
low_params= Q_table['low_params']
low_tree_dist_bin_size = low_params[0]
low_relative_dist_bin_size = low_params[1]
low_velocity_bucket_size = low_params[2]
high_tree_dist_bin_size = high_params[0]
high_relative_dist_bin_size = high_params[1]
high_velocity_bucket_size = high_params[2]


class LowGTableTrained(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        #new class_vars
        self.gravity = 1
        self.tick_number = 0
        self.Q = Q_table
        self.firstState = None
        self.counter = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # Reset new class_vars
        self.gravity = 1
        self.tick_number = 0

        # Increment counter
        self.counter+=1

    def action_callback(self, state):

        new_state  = self.simplifyState(state)

        if new_state not in self.Q:
            self.Q[new_state] = [0,0]

        #Choose best action
        Q_state = self.Q[new_state]
        if Q_state[0] == Q_state[1]:

           new_action = 0

        elif  Q_state[1] > Q_state[0]:
            new_action = 1
        else:
            new_action = 0

        # But if first first tick, dont jump, must calculate gravity first
        if self.tick_number == 0:
            new_action = 0

        # Calculate gravity
        if self.tick_number == 1:
            self.gravity = int(abs(state['monkey']['vel'] - self.firstState['monkey']['vel']))

        # Finished iteration, update vars
        if self.tick_number == 0:
            self.firstState = state
        self.last_action = new_action
        self.last_state  = new_state
        self.tick_number += 1
        return self.last_action

    def simplifyState(self, state):
        monkey_bottom = state['monkey']['bot']
        tree_bottom = state['tree']['bot']
        dist_difference = monkey_bottom - tree_bottom
        tree_dist = state['tree']['dist']
        monkey_vel = state['monkey']['vel']

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

    def discretizeVel(self, value, size):
        if size == 'Binary':
            # find sign of velocity
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



