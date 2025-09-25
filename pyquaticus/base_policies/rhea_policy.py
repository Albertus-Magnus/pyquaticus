from typing import Union

import numpy as np

import copy
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
        super().__init__(agent_id, env)
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

            def _prepare_for_deepcopy(self, envi): #TODO delete if this doesnt work
                """
                Remove or null out non-picklable rendering objects (pygame.Surface, windows, renderers)
                and return a dictionary with saved objects so they can be restored after deepcopy.
                Call _restore_after_deepcopy(saved) once done.
                """
                saved = {"attrs": {}, "player_attrs": {}}
                try:
                    import pygame
                    Surface = pygame.Surface
                except Exception:
                    Surface = None

                # top-level attributes
                for name, val in list(vars(envi).items()):
                    try:
                        if Surface is not None and isinstance(val, Surface):
                            saved["attrs"][name] = val
                            setattr(envi, name, None)
                        # common render handles by name
                        elif name in ("renderer", "window", "screen", "clock", "pygame_background_img"):
                            saved["attrs"][name] = val
                            setattr(envi, name, None)
                    except Exception:
                        pass

                # player-level attributes
                players = getattr(envi, "players", None) or {}
                for pid, player in players.items():
                    saved["player_attrs"].setdefault(pid, {})
                    for attr in ("pygame_agent", "renderer", "window", "surface", "image"):
                        try:
                            if hasattr(player, attr):
                                v = getattr(player, attr)
                                if Surface is not None and isinstance(v, Surface):
                                    saved["player_attrs"][pid][attr] = v
                                    setattr(player, attr, None)
                                elif attr in ("renderer", "window") and v is not None:
                                    # save and null out generic renderer/window references
                                    saved["player_attrs"][pid][attr] = v
                                    setattr(player, attr, None)
                        except Exception:
                            pass

                return saved

            def _restore_after_deepcopy(self, envi, saved): #TODO delete if this doesnt work
                """Restore things saved by _prepare_for_deepcopy."""
                if not saved:
                    return
                for name, val in saved.get("attrs", {}).items():
                    try:
                        setattr(envi, name, val)
                    except Exception:
                        pass
                players = getattr(envi, "players", None) or {}
                for pid, attrs in saved.get("player_attrs", {}).items():
                    player = players.get(pid, None)
                    if player is None:
                        continue
                    for attr, val in attrs.items():
                        try:
                            setattr(player, attr, val)
                        except Exception:
                            pass

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
                    self._prepare_for_deepcopy(self._py_env)
                    saved = self._start_env = copy.deepcopy(self._py_env)
                    self._restore_after_deepcopy(self._py_env, saved)
                    print(self._py_env)
                    print("marker 87")
                except Exception:
                    self._start_env = self._py_env
            def get_random_action(self):
                space = self._action_space or self._try_agent_action_space()
                if space is not None:
                    try:
                        return space.sample()
                    except Exception:
                        pass
                if self._continuous:
                    import numpy as _np
                    speed = _np.random.random() * self._max_speed
                    heading = (_np.random.random() * 360.0) - 180.0
                    return (speed, heading)
                else:
                    import numpy as _np
                    return int(_np.random.randint(0, 15))
            def evaluate_rollout(self, solutions, discount_factor=None, ignore_frames=0):
                scores = []
                for sol in solutions:
                    try:
                        saved = self._prepare_for_deepcopy(self._start_env)
                        sim_env = copy.deepcopy(self._start_env)
                        self._restore_after_deepcopy(self._start_env, saved)
                        #print(self._start_env) #is type None
                        #print(sim_env) #is type None
                        #return -1
                        # usually no exception here (testing result)
                    except Exception:
                        # Is not usually thrown (testing result)
                        print("Exception during deepcopy of start_env, using current env as start_env.")
                        saved = self._prepare_for_deepcopy(self._py_env)
                        sim_env = copy.deepcopy(self._py_env)
                        self._restore_after_deepcopy(self._py_env, saved)
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

        rollout_actions_length = 100
        mutation_probability = 0.2
        num_evals = 1600
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
        # Update the state based on this observation (remain from base_combined, still purpose?)
        #self.update_state(obs, info)

        # compute next action should be already implemented by _get_next_action
        action = self.rhea._get_next_action()
        if action is None:
            print("RHEA returned None action, falling back to ultra defensive")
        else:
            return action

        #fallback to ultra defensive:
        return self.action_from_vector(None, 0)

    # def random_defense_action(self, enem_positions): #Perhaps this reminder of base_combined can be deleted...
    #     """
    #     (--Remains from base_combined--)
    #     Randomly compute an action that steers the agent to it's own side of the field and sometimes
    #     towards its flag.
    #     """
    #     if np.random.random() < 0.25:
    #         # go to random point on segment between my flag and scrimmage line
    #         t = np.random.random()
    #         goal_vec = (1 - t) * self.my_flag_loc + t * self.midpoint_local

    #     else:
    #         near_enemy_dist = np.inf
    #         nearest_enemy = None
    #         for en in enem_positions:
    #             temp_enem_dist = en[0]
    #             if temp_enem_dist < near_enemy_dist:
    #                 near_enemy_dist = temp_enem_dist
    #                 nearest_enemy = en
    #         assert nearest_enemy is not None
    #         if np.random.random() < 0.5:
    #             goal_vec = rel_bearing_to_local_unit_rect(nearest_enemy[1])
    #         else:
    #             own_flag_dist = self.my_flag_distance
    #             if own_flag_dist > self.flag_keepout + 2.0:
    #                 goal_vec = rel_bearing_to_local_unit_rect(self.my_flag_bearing)
    #             else:
    #                 # want random point on segment between my flag and scrimmage line, but at least <defensiveness> meters from scrimmage line
    #                 t = np.random.random()
    #                 unit_vec = unit_vect_between_points(
    #                     self.midpoint_local, self.my_flag_loc
    #                 )
    #                 endpoint = (
    #                     self.midpoint_local
    #                     + min(
    #                         self.defensiveness,
    #                         dist(self.midpoint_local, self.my_flag_loc),
    #                     )
    #                     * unit_vec
    #                 )
    #                 goal_vec = (1 - t) * self.my_flag_loc + t * endpoint

    #     if not self.on_sides:
    #         goal_vec = goal_vec + get_avoid_vect(self.opp_team_pos, avoid_threshold=15)

    #     if self.mode == "hard":
    #         return self.action_from_vector(goal_vec, 1)
    #     else:
    #         return self.action_from_vector(goal_vec, 0.5)

    # def update_state(self, obs, info: dict[str, dict]) -> None:
    #     """
    #     (--Remains from base_combined--)
    #     Method to convert the gym obs and info into data more relative to the
    #     agent.

    #     Note: all rectangular positions are in the ego agent's local coordinate frame.
    #     Note: all bearings are relative, measured in degrees clockwise from the ego agent's heading.

    #     Args:
    #         obs: observation from gym
    #         info: info from gym
    #     """

    #     global_state = info[self.id]["global_state"]
    #     if not isinstance(global_state, dict):
    #         global_state = self.state_normalizer.unnormalized(global_state)

    #     self.on_sides = global_state[(self.id, "on_side")]
    #     self.has_flag = global_state[(self.id, "has_flag")]

    #     # Copy the polar positions of each agent, separated by team and get their tag status
    #     self.opp_team_pos = []
    #     self.my_team_pos = []
    #     self.opp_team_tag = []
    #     self.opp_team_has_flag = False
    #     for id in self.teammate_ids:
    #         if id != self.id:
    #             distance = dist(
    #                 global_state[(self.id, "pos")], global_state[(id, "pos")]
    #             )
    #             bearing = angle180(
    #                 global_rect_to_abs_bearing(
    #                     global_state[(id, "pos")] - global_state[(self.id, "pos")]
    #                 )
    #                 - global_state[(self.id, "heading")]
    #             )
    #             self.my_team_pos.append(np.array((distance, bearing)))
    #     for id in self.opponent_ids:
    #         distance = dist(global_state[(self.id, "pos")], global_state[(id, "pos")])
    #         bearing = angle180(
    #             global_rect_to_abs_bearing(
    #                 global_state[(id, "pos")] - global_state[(self.id, "pos")]
    #             )
    #             - global_state[(self.id, "heading")]
    #         )
    #         self.opp_team_pos.append(np.array((distance, bearing)))
    #         self.opp_team_has_flag = (
    #             self.opp_team_has_flag or global_state[(id, "has_flag")]
    #         )
    #         self.opp_team_tag.append(global_state[(id, "is_tagged")])
    #     team_str = self.team.name.lower().split("_")[0]
    #     opp_str = "red" if team_str == "blue" else "blue"
    #     self.my_flag_distance = dist(
    #         global_state[(self.id, "pos")], global_state[team_str + "_flag_pos"]
    #     )
    #     self.my_flag_bearing = angle180(
    #         global_rect_to_abs_bearing(
    #             global_state[team_str + "_flag_pos"] - global_state[(self.id, "pos")]
    #         )
    #         - global_state[(self.id, "heading")]
    #     )
    #     self.my_flag_loc = dist_rel_bearing_to_local_rect(
    #         self.my_flag_distance, self.my_flag_bearing
    #     )
    #     self.opp_flag_distance = dist(
    #         global_state[(self.id, "pos")], global_state[opp_str + "_flag_pos"]
    #     )
    #     self.opp_flag_bearing = angle180(
    #         global_rect_to_abs_bearing(
    #             global_state[opp_str + "_flag_pos"] - global_state[(self.id, "pos")]
    #         )
    #         - global_state[(self.id, "heading")]
    #     )
    #     self.opp_flag_loc = dist_rel_bearing_to_local_rect(
    #         self.opp_flag_distance, self.opp_flag_bearing
    #     )

    #     self.my_team_density, self.opp_team_density = self.get_team_density(
    #         self.my_team_pos, self.opp_team_pos
    #     )

    #     self.midpoint_local = global_rect_to_local_rect(
    #         self.midpoint_global,
    #         global_state[(self.id, "pos")],
    #         global_state[(self.id, "heading")],
    #     )

    # def action_from_vector(self, vector, desired_speed_normalized):
    #     """
    #     (--Remains from base_combined--)
    #     Convert a desired vector in local rectangular coordinates and a desired speed
    #     (0 to 1) into either a continuous or discrete action.
    #     """
    #     if desired_speed_normalized == 0:
    #         if self.continuous:
    #             return (0, 0)
    #         else:
    #             return -1
    #     rel_bearing = local_rect_to_rel_bearing(vector)
    #     if self.continuous:
    #         return (desired_speed_normalized * self.max_speed, rel_bearing)
    #     elif desired_speed_normalized == 0.5:
    #         if 1 >= rel_bearing >= -1:
    #             return 12
    #         elif rel_bearing < -1:
    #             return 14
    #         elif rel_bearing > 1:
    #             return 10
    #     elif desired_speed_normalized == 1:
    #         if 1 >= rel_bearing >= -1:
    #             return 4
    #         elif rel_bearing < -1:
    #             return 6
    #         elif rel_bearing > 1:
    #             return 2

