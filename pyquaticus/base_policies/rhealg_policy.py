from copy import deepcopy
import random
from typing import Any, Union
import numpy as np
# I Copied rhea code into a subfolder of the current folder for easy access in imports
from .RollingHorizonEvolAlg.environment import Environment
from .RollingHorizonEvolAlg.rhea import RollingHorizonEvolutionaryAlgorithm
# (special rhea import is thus no longer needed)
from pyquaticus import pyquaticus_v0
from pyquaticus.envs.pyquaticus import PyQuaticusEnv#, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
#Start of special rhea import
# try the normal absolute import first; if it fails, add project root and retry
# try:
#     from RollingHorizonEvolutionaryAlgorithm.RollingHorizonEA.rhea import RollingHorizonEvolutionaryAlgorithm
# except ModuleNotFoundError:
#     import sys, os, importlib
#     # compute project root relative to this file (../../.. -> project root)
#     _this_dir = os.path.dirname(__file__)
#     _project_root = os.path.abspath(os.path.join(_this_dir, "..", "..", ".."))
#     if _project_root not in sys.path:
#         sys.path.insert(0, _project_root)
#     # retry import (let any exception propagate if it still fails)
#     RollingHorizonEvolutionaryAlgorithm = importlib.import_module(
#         "RollingHorizonEvolutionaryAlgorithm.RollingHorizonEA.rhea"
#     ).RollingHorizonEvolutionaryAlgorithm
# #End of special rhea import

# RHEA parameters (adjust here globally)
rollout_actions_length = 3#100
mutation_probability = 0.3
num_evals = 2#100
# End of RHEA parameters

class RHEA_Agent(BaseAgentPolicy):
    """
    Copied from BaseAgentPolics and modified to implement RHEA agent.
    """

    def __init__(
        self,
        agent_id: str,
        rhea_env,#: RHEA_Environment,#Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        suppress_numpy_warnings=True,
        continuous: bool = False,
    ):
        super().__init__(agent_id, env)
        # initialize rhea heuristic class
        self.rhea = RollingHorizonEvolutionaryAlgorithm(
            rollout_actions_length, 
            rhea_env, 
            mutation_probability, 
            num_evals, 
            use_shift_buffer=True,
            flip_at_least_one=True, 
            discount_factor=None, 
            ignore_frames=0
        ) # If this RHEA is adjusted properly for multiple teammates it might need to be initialized centrally.
        self.rhea_env = rhea_env
        # End of __init__

    def compute_action(self, obs, info: dict[str, dict]) -> Any:
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and relative bearing.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """
        action = self.rhea._get_next_action()
        return action
        # End of compute_action

"""
This Class inherits from the Rolling Horizon Evolutionary Algorithm Environment interface.
"""
class RHEA_Environment(Environment):
    
    def __init__(self, env):#agent_id, team, obs, info):
        # self.agent_id = agent_id #old stuff, delete later?
        # self.team = team
        # self.obs = obs
        # self.info = info
        #Adding action_map to this env for easy access:
        self.action_map = []
        for spd in [1.0, 0.5]:
            for hdg in range(180, -180, -45):
                self.action_map.append([spd, hdg])
        # add a none action
        self.action_map.append([0.0, 0])
        # End of action_map
        """ Initialize a copy of the pyquaticus environment """
        config_dict = {}
        config_dict["max_time"] = 600.0
        config_dict["max_score"] = 100
        #config_dict["render_agent_ids"] = True
        config_dict["dynamics"] = ["si", "si", "si", "si", "si", "si"]
        config_dict["sim_speedup_factor"] = 3

        self.env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict,render_mode=None)
        # term_g = {'agent_0':False,'agent_1':False,'agent_2':False}
        # truncated_g = {'agent_0':False,'agent_1':False,'agent_2':False}
        # term = term_g
        # trunc = truncated_g
        # The following line assumes the env is brand new and also just reset.
        reset_opts = {'normalize_obs': False, 'normalize_state': False}
        #obs, info = 
        self.env.reset(options=reset_opts) #is this necessary? maybe it is just done to get obs, info but the False above make it so it doesn't do anything...

        # temp_captures = env.state["captures"]
        # temp_grabs = env.state["grabs"]
        # temp_tags = env.state["tags"]
        # End of env copy init
        # End of __init__

    def perform_action(self, action):
        # I implement this to keep the environment up to date at each real-environment step.
        # action contains multiple actions, but step is already taking care of it.
        self.env.step(action)

    # I may need to change the input to x solutions and y solution. Then adjust rhea algorithm to handle the opponent moves itself.
    def evaluate_rollout(self, solutions, discount_factor=0, ignore_frames=0):
        """
        Used in rhea.py as:
        mutated_scores = self._environment.evaluate_rollout(
            candidate_solutions, 
            self._discount_factor,
            self._ignore_frames
        )
        Should return a numpy array of scores (reward maxing?).
        Input is multiple solutions? Yes, probably.


        --- Example implementation (different game): ---

        n_evals = solutions.shape[0]

        state_copies = np.repeat(np.expand_dims(self._game_state, 0), n_evals, axis=0)

        for state_copy, solution in zip(state_copies, solutions):
            for action in solution:
                state_copy[action[0]] += action[1]

        return self._score_states(state_copies)
        """
        n_evals = solutions.shape[0]#numpy list/array shape function?
        # I need to make copies of the environment, this is my version of game state:
        state_copies = [deepcopy(self.env) for _ in range(n_evals)] #TODO test if deepcopy works for Pyquaticus env
        # Then I need to apply the solutions to the copies.:
        scores = np.zeros(n_evals)
        i = 0
        for state_copy, solution in zip(state_copies, solutions):
            #do I need to do this multiple times for each opponent?:
            for action in solution:
                #state_copy[action[0]] += action[1] probably old code from the example, delete
                # Provisional solution for testing: Just one random solution per opponent, as in RHGA with low budget...
                zero = [0.0, 0]
                one = action
                two = [0.0, 0]
                #three = np.array([self.get_random_action() for _ in range(rollout_actions_length)])
                #this is incorrect, is entire solution where only one action is needed...
                #four = np.array([self.get_random_action() for _ in range(rollout_actions_length)])
                #five = np.array([self.get_random_action() for _ in range(rollout_actions_length)])
                #Provisorisch: jeweils eine zufallsaktion:
                three = self.get_random_action()
                four = self.get_random_action()
                five = self.get_random_action()

                #TODO I may need to compute action for every other agent in the game here so the plan always uses correct other agents. Though for the competition it would be unfair to have an agent know what the other agents do... So how do We implement this? - also rhea would be best, but now doing a deadlock...
                # F*ck... -we move this to rhea TODO        #one is the rhea agent, zero and two are None (too expensive). The rest are rhea estimates of enemies :/
                obs, reward, term, trunc, info = self.env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five}) #TODO we have only one action so far, what do the other agents do in our plan?
                scores[i] += reward #This is assuming the reward is per step. If the reward is a score that keeps its pos/neg values from previous steps only the last score needs to be remembered/added.
            i += 1
        # The final(?)/summed(?) reward of the solution has to be stored in a list and returned:
        return scores #currently a sum of all steps


    def get_random_action(self):
        #return a random action based on the ACTION_MAP in config.py (available in self as well)
        return random.choice(self.action_map)

    def is_game_over(self):
        # this is only necessary if rhea.run() is used, which it isn't in the current context
        if self.env.__getattribute__("term") or self.env.__getattribute__("trunc"):
            return True
        return False

    def get_current_score(self):
        # this is only necessary if rhea.run() is used, which it isn't in the current context
        raise NotImplementedError #TODO (might still need it?)

    def ignore_frame(self):
        # this is only necessary if rhea.run() is used, which it isn't in the current context
        raise NotImplementedError 
    
    # End of RHEA_Environment
