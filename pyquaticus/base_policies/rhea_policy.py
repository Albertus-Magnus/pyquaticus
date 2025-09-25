import random
from typing import Union

import numpy as np

import copy
from pyquaticus import pyquaticus_v0
import pyquaticus.base_policies.base_attack as attack_policy
import pyquaticus.base_policies.base_defend as defend_policy
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.base_policies.utils import (dist_rel_bearing_to_local_rect,
                                            get_avoid_vect,
                                            global_rect_to_abs_bearing,
                                            global_rect_to_local_rect,
                                            local_rect_to_rel_bearing,
                                            rel_bearing_to_local_unit_rect,
                                            unit_vect_between_points)
from pyquaticus.config import config_dict_std
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180, dist, line_intersection
from RollingHorizonEA.rhea import RollingHorizonEvolutionaryAlgorithm
#RollingHorizonEvolutionaryAlgorithm.RollingHorizonEA.rhea import RollingHorizonEvolutionaryAlgorithm

#MODES = {"easy", "medium", "hard", "nothing"}


class RHEA_CTF_Agent(BaseAgentPolicy):
    """CTF agent policy utilizing the RHE Algorithm to compute the next action."""

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        flag_keepout: float = config_dict_std["flag_keepout"],
        catch_radius: float = config_dict_std["catch_radius"],
        continuous: bool = False,
        #defensiveness: float = 20.0,
    ):
        super().__init__(agent_id, env, continuous)
        # Testing if I can turn a fresh initiate of env into a copy of the original env (but deepcopyable)
        config_dict = {}
        config_dict["max_time"] = 600.0
        config_dict["max_score"] = 100
        config_dict["render_agent_ids"] = True
        config_dict["dynamics"] = ["si", "si"]#["si", "si", "si", "si", "si", "si"]
        config_dict["sim_speedup_factor"] = 3
        temp_env = pyquaticus_v0.PyQuaticusEnv(team_size=1, config_dict=config_dict,render_mode=None) #best idea I've ever had
        reset_opts = {'normalize_obs': False, 'normalize_state': False}
        obs, info = temp_env.reset(options=reset_opts)
        env =  temp_env #changes everything (super happy about that, but why no self. here?)
        self.env = temp_env #doesnt change anything
        # End of initialize copyable env #TODO check if env.step is done or has to be added

        self.state_normalizer = env.global_state_normalizer
        self.walls = env._walls[self.team.value]
        self.max_speed = env.max_speeds[env.players[self.id].idx]
        #self.defensiveness = defensiveness
        self.continuous = continuous
        self.flag_keepout = flag_keepout

        scrimmage_line = env.scrimmage_coords
        flag_line = np.array(
            (env.flag_homes[Team.RED_TEAM], env.flag_homes[Team.BLUE_TEAM])
        )
        self.midpoint_global = line_intersection(scrimmage_line, flag_line)

        # Add RHEA instance as property
        # Wrapper class to make the pyquaticus env compatible with RHEA:
        class SingleAgentRHEAEnv: #TODO does this work? -preliminary answer no because it tries to copy 
            def __init__(self, py_env, agent_id, continuous, max_speed):
                #Adding action_map to this env for easy access:
                self.action_map = []
                for spd in [1.0, 0.5]:
                    for hdg in range(180, -180, -45):
                        self.action_map.append([spd, hdg])
                # add a none action
                self.action_map.append([0.0, 0.0])
                #End of action_map
                self._py_env = py_env
                self._agent_id = agent_id
                self._continuous = continuous
                self._max_speed = max_speed
                self._action_space = self._try_agent_action_space() #perhaps this could be changed into a descrete/continuous check?
                #self._start_env = None #was wrong? set same as py_env for now, maybe is not needed as its own variable
                self._start_env = py_env
                # Testing if I can make env conform to deepcopy by removing unpicklable grafics:
                self._start_env.renderer = None
                self._start_env.window = None
                self._start_env.pygame_background_img = None
                # (just to be safe, possible redundancy):
                self._py_env.renderer = None
                self._py_env.window = None
                self._py_env.pygame_background_img = None #TODO does this affect the actual env?
                # testing that...
                if self._action_space is not None:
                    try:
                        self._other_action = self._action_space.sample()
                    except Exception:
                        self._other_action = (0.0, 0.0) if continuous else -1
                else:
                    self._other_action = (0.0, 0.0) if continuous else -1

            def _try_agent_action_space(self):
                try:
                    sp = getattr(self._py_env, "action_space", None)
                    if sp is not None:
                        try:
                            return sp[self._agent_id]
                        except Exception:
                            return sp
                    players = getattr(self._py_env, "players", None)
                    if players is not None and self._agent_id in players:
                        return getattr(players[self._agent_id], "action_space", None)
                except Exception:
                    pass
                return None
            def set_start_state(self, obs, info):
                try:
                    #self._prepare_for_deepcopy(self._py_env)
                    saved = self._start_env = copy.deepcopy(self._py_env)
                    #self._restore_after_deepcopy(self._py_env, saved)
                    print(self._py_env)
                    print("marker 87")
                except Exception:
                    self._start_env = self._py_env
            def get_random_action(self):
                #return a random action based on the ACTION_MAP in config.py (available in self as well)
                return copy(random.choice(self.action_map))
            # def get_random_action(self):#old version, not specific to pyquaticus
            #     space = self._action_space or self._try_agent_action_space()
            #     if space is not None:
            #         try:
            #             return space.sample()
            #         except Exception:
            #             pass
            #     if self._continuous:
            #         import numpy as _np
            #         speed = _np.random.random() * self._max_speed
            #         heading = (_np.random.random() * 360.0) - 180.0
            #         return (speed, heading)
            #     else:
            #         import numpy as _np
            #         return int(_np.random.randint(0, 15))
            def evaluate_rollout(self, solutions, discount_factor=None, ignore_frames=0):
                scores = []
                for sol in solutions:
                    try:
                        #saved = self._prepare_for_deepcopy(self._start_env)
                        sim_env = copy.deepcopy(self._start_env)
                        #self._restore_after_deepcopy(self._start_env, saved)
                        #print(self._start_env) #is type None
                        #print(sim_env) #is type None
                        #return -1
                        # usually no exception here (testing result)
                    except Exception:
                        # Is not usually thrown (testing result)
                        print("Exception during deepcopy of start_env, using current env as start_env.")
                        #saved = self._prepare_for_deepcopy(self._py_env)
                        sim_env = copy.deepcopy(self._py_env)
                        #self._restore_after_deepcopy(self._py_env, saved)
                    total_reward = 0.0
                    discount = 1.0
                    for action in sol:
                        action_dict = {self._agent_id: action}
                        try:
                            obs, reward, terminated, truncated, info = sim_env.step(action_dict)
                        except Exception as e: #'NoneType' object has no attribute 'step'
                            print(e)
                            print("Exception during RHEA env step, unable to get reward from .step()") #HERE
                            try:
                                obs, reward, done, info = sim_env.step(action_dict)
                                terminated = done.get(self._agent_id, False) if isinstance(done, dict) else done
                                truncated = False
                            except Exception as e_: #'NoneType' object has no attribute 'step'
                                terminated = True
                                truncated = True
                                print(e_)
                                print("Reward 0.0 because .step() threw error.") #THEN HERE
                                reward = {self._agent_id: 0.0}
                        r = 0.0
                        if isinstance(reward, dict):
                            print("marker 532") #THEN HERE
                            r = float(reward.get(self._agent_id, 0.0))
                            print("r = ", r)
                        else:
                            print("marker 165")
                            r = float(reward)
                        total_reward += discount * r
                        if discount_factor is not None:
                            discount *= discount_factor
                        if terminated or truncated:
                            break
                    scores.append(total_reward)
                import numpy as _np
                #print("Scores of evaluations:", scores)#TODO delete after testing
                return _np.array(scores)
            def perform_action(self, action):
                action_dict = {self._agent_id: action}
                try:
                    self._py_env.step(action_dict)
                except Exception:
                    pass
            def is_game_over(self):
                try:
                    return getattr(self._py_env, "game_over", False)
                except Exception:
                    return False
            def get_current_score(self):
                return 0.0

        rollout_actions_length = 2#100
        mutation_probability = 0.2
        num_evals = 3#1600
        self.rhea = RollingHorizonEvolutionaryAlgorithm(
            rollout_actions_length,
            SingleAgentRHEAEnv(env, self.id, continuous, self.max_speed),
            mutation_probability,
            num_evals
        )

    def compute_action(self, obs, info): #TODO implement rhea
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and heading error.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """
        # compute next action should be already implemented by _get_next_action
        action = self.rhea._get_next_action()
        print("RHEA action:", action)
        if action is None:
            print("RHEA returned None action, falling back to ultra defensive")
        else:
            return action
        #fallback to ultra defensive:
        return self.action_from_vector(None, 0)


    def action_from_vector(self, vector, desired_speed_normalized):
        """
        (--Remains from base_combined--)
        Convert a desired vector in local rectangular coordinates and a desired speed
        (0 to 1) into either a continuous or discrete action.
        """
        if desired_speed_normalized == 0:
            if self.continuous:
                return (0, 0)
            else:
                return -1
        rel_bearing = local_rect_to_rel_bearing(vector)
        if self.continuous:
            return (desired_speed_normalized * self.max_speed, rel_bearing)
        elif desired_speed_normalized == 0.5:
            if 1 >= rel_bearing >= -1:
                return 12
            elif rel_bearing < -1:
                return 14
            elif rel_bearing > 1:
                return 10
        elif desired_speed_normalized == 1:
            if 1 >= rel_bearing >= -1:
                return 4
            elif rel_bearing < -1:
                return 6
            elif rel_bearing > 1:
                return 2

