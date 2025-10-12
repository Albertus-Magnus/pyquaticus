from copy import deepcopy
import random
from typing import Any, Union
import numpy as np

from pyquaticus.utils.rewards import triple_aggressive_rew, triple_caps_and_grabs
# I Copied rhea code into a subfolder of the current folder for easy access in imports
from .RollingHorizonEvolAlg.environment import Environment
from .RollingHorizonEvolAlg.multirhea import MultipleRollingHorizonEvolutionaryAlgorithm
# (special rhea import is thus no longer needed)
from pyquaticus import pyquaticus_v0
from pyquaticus.envs.pyquaticus import PyQuaticusEnv#, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.base_policies.base_policy import BaseAgentPolicy

# RHEA parameters (adjust here globally)
rollout_actions_length = 5#100
mutation_probability = 0.5
num_evals = 12#100
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
        self.rhea = MultipleRollingHorizonEvolutionaryAlgorithm(
            (rollout_actions_length*3), #creates solutions for all 3 agents, ABC are concatenated and treated accordingly in compute_action and evaluate_rollout
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
        action = self.rhea._get_next_action() #actually we need to make sure get_next_action already returns 3 actions TODO
        # action is actually 3 actions for 3 agents:
        #parts = np.array_split(action, 3)#not entire solutions, so we need only pass them
        return action #list(3x) of lists(rollout-length) of actions
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
        """ Initialize a copy of the pyquaticus environment """
        config_dict = {}
        config_dict["max_time"] = 600.0
        config_dict["max_score"] = 100
        #config_dict["render_agent_ids"] = True
        config_dict["dynamics"] = ["si", "si", "si", "si", "si", "si"]
        config_dict["sim_speedup_factor"] = 3

        self.env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict, reward_config={'agent_1': triple_caps_and_grabs},render_mode=None)
        # The following line assumes the env is brand new and also just reset.
        reset_opts = {'normalize_obs': False, 'normalize_state': False}
        #obs, info = 
        self.env.reset(options=reset_opts) 
        #Adding action_map to this env for easy access:
        self.action_map = []
        # for spd in [1.0, 0.5]: #speed will always be 1.0. Less speed is always never optimal and this counters computational expense
        spd = 1.0
        spd = self.env.max_speeds[0]
        #print("max speed: ",spd)
        for hdg in range(180, -180, -45): #8 different directions possible
            self.action_map.append([spd, hdg])
        # add a none action
        self.action_map.append([0.0, 0])
        # End of action_map

    def perform_action(self, action, statee):
        # I implement this to keep the environment up to date at each real-environment step.
        # action contains multiple actions, but step is already taking care of it.
        self.env.step(action) #should i remove this line? No. then its (maybe) worse
        # print(self.env.state['agent_position'])
        self.env.state = deepcopy(statee) #this is the important line, probably the only one I need?
        #THE 3 LINES BELOW ARE NECESSARY TO UPDATE STATE:
        self.env._set_player_attributes_from_state()
        self.env._set_flag_attributes_from_state()
        self.env._set_game_events_from_state()

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
        state_copies = [deepcopy(self.env) for _ in range(n_evals)]
        # Then I need to apply the solutions to the copies.:
        scores = np.zeros(n_evals)
        i = 0
        #solutionthrice = np.array_split(solution, 3)
        for state_copy, solution in zip(state_copies, solutions):
            #do I need to do this multiple times for each opponent?:
            solutri = np.array_split(solution, 3)
            #print("solutri: ",solutri)
            # for index in range(len(solutri[0]) - 1):
            for index in range(len(solutri[0])):
                # RHGA with low computational budget: Just one random solution per opponent.^
                zero = solutri[0][index] #Should be ith action in the first agents solution block
                one = solutri[1][index] #Should be ith action in the seccond agents solution block
                two = solutri[2][index] #Should be ith action in the third agents solution block
                three = self.get_random_action()
                four = self.get_random_action()
                five = self.get_random_action()
                obs, reward, term, trunc, info = state_copy.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five}) #TODO we have only one action so far, what do the other agents do in our plan?
                # print("Reward returned: ",reward['agent_1'])
                scores[i] += reward['agent_1'] #This is assuming the reward is per step.
            i += 1
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
    
    def splitlist(l,n):
    # split list l into n parts, could use this instead of splitarray if there are type mismatches
        if l: 
            p = len(l) if n < 1 else len(l) // n
            p = p if p > 0 else 1
            for i in range(0, len(l), p):
                yield l[i:i+p]
        else:
            # empty list split returns empty list
            yield [] 

    
    # End of RHEA_Environment
