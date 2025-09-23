import logging
import numpy as np

from RollingHorizonEA import RollingHorizonEvolutionaryAlgorithm
from RollingHorizonEA.environment import Environment

class PyquaticGame(Environment):
    '''
    The aim of this game is to capture the flag more often than the enemy team.

    An action is a direction to turn in and a speed to move at.

    Capturing the flag is done automatically, as well as tagging enemies.

    THIS FILE IS NOT SERVING A PURPOSE! MAYBE DELETE LATER...
    '''

    def __init__(self):
        super(PyquaticGame, self).__init__("Pyquaticus Game")

    def _score_states(self, state):

    def evaluate_rollout(self, solutions, discount_factor=0, ignore_frames=0):

    def perform_action(self, action):

    def get_random_action(self):

    def is_game_over(self):

    def get_current_score(self):

    def ignore_frame(self):


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    num_dims = 600
    m = 50
    num_evals = 50
    rollout_length = 100
    mutation_probability = 0.1

    # Set up the problem domain as one-max problem
    environment = PyquaticGame(num_dims, m)

    rhea = RollingHorizonEvolutionaryAlgorithm(rollout_length, environment, mutation_probability, num_evals)

    rhea.run()

