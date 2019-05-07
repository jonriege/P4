import pygame as pg
from SwingyMonkey import SwingyMonkey
import argparse
import matplotlib.pyplot as plt

from SimpleNN import SimpleNNLearner, SimpleNNTrained
from ReplayNN import ReplayNNLearner, ReplayNNTrained
from HighGTableLearner import HighGTableLearner
from HighGTableTrained import HighGTableTrained
from LowGTableLearner import LowGTableLearner
from LowGTableTrained import LowGTableTrained


AGENT_DICT = {
    'simple_nn': {True: SimpleNNLearner, False: SimpleNNTrained},
    'replay_nn': {True: ReplayNNLearner, False: ReplayNNTrained},
    'low_g_table': {True: LowGTableLearner, False: LowGTableTrained},
    'high_g_table': {True: HighGTableLearner, False: HighGTableTrained}
}


def run_games(agent, epochs, tick_length):
    history = []
    for epoch in range(epochs):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,
                             text=f"Epoch {epoch}",
                             tick_length=tick_length,
                             action_callback=agent.action_callback,
                             reward_callback=agent.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        history.append(swing.score)

        # Reset the state of the learner.
        agent.reset()

        if epoch > 0 and epoch % 100 == 0:
            print(max(history[-100:]))
            print(sum(history[-100:])/100.0)
    pg.quit()
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", help="Possible values: simple_nn, replay_nn, low_g_table, high_g_table", type=str)
    parser.add_argument("-t", "--train", help="True: Train new agent, False: Use saved agent", type=bool, default=False)
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=400)
    parser.add_argument("-tl", "--tick_length", help="Tick length in ms", type=int, default=50)
    args = parser.parse_args()

    agent = AGENT_DICT[args.agent][args.train]()
    history = run_games(agent, args.epochs, args.tick_length)

    fig, ax = plt.subplots()
    ax.plot(history)
    plt.show()
