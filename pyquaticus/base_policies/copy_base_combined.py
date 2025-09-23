# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

from typing import Union

import numpy as np

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

# NEW imports for RHEA integration
import copy
import numpy as _np
#try:
from RollingHorizonEA.rhea import RollingHorizonEvolutionaryAlgorithm
#except Exception:
	# Try alternative import path if package layout differs #TODO this exception can be removed
#	from RollingHorizonEvolutionaryAlgorithm.RollingHorizonEA.rhea import RollingHorizonEvolutionaryAlgorithm

MODES = {"easy", "medium", "hard", "nothing", "rhea"}


class Heuristic_CTF_Agent(BaseAgentPolicy):
    """Combined attack and defense policy accounting for enemy positions before deciding to attack/defend."""

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        flag_keepout: float = config_dict_std["flag_keepout"],
        catch_radius: float = config_dict_std["catch_radius"],
        continuous: bool = False,
        mode: str = "easy",
        defensiveness: float = 20.0,
    ):
        super().__init__(agent_id, env)
        self.state_normalizer = env.global_state_normalizer
        self.walls = env._walls[self.team.value]
        self.max_speed = env.max_speeds[env.players[self.id].idx]
        self.set_mode(mode)
        self.defensiveness = defensiveness
        self.continuous = continuous
        self.flag_keepout = flag_keepout
        if mode != "rhea":
            self.base_attacker = attack_policy.BaseAttacker(
                self.id,
                env,
                continuous,
                mode,
            )
            self.base_defender = defend_policy.BaseDefender(
                self.id,
                env,
                flag_keepout,
                catch_radius,
                continuous,
                mode,
            )
        else:
            self.base_attacker = None
            self.base_defender = None

        scrimmage_line = env.scrimmage_coords
        flag_line = np.array(
            (env.flag_homes[Team.RED_TEAM], env.flag_homes[Team.BLUE_TEAM])
        )
        self.midpoint_global = line_intersection(scrimmage_line, flag_line)

        ######START of cpilot edit
        # If using RHEA, create a small adapter that implements the minimal env API RHEA expects
        self.rhea = None
        if self.mode == "rhea":
            # minimal SingleAgent wrapper
            class SingleAgentRHEAEnv:
                def __init__(self, py_env, agent_id, teammate_ids, opponent_ids, continuous, max_speed):
                    self._py_env = py_env
                    self._agent_id = agent_id
                    self._teammate_ids = list(teammate_ids)
                    self._opponent_ids = list(opponent_ids)
                    self._continuous = continuous
                    self._max_speed = max_speed
                    # snapshot that will be used as rollout starting point
                    self._start_env = None
                    # default other agents action (stationary)
                    if continuous:
                        self._other_action = (0.0, 0.0)
                    else:
                        self._other_action = -1

                def set_start_state(self, obs, info):
                    # snapshot the current env for fast rollouts (deepcopy)
                    try:
                        self._start_env = copy.deepcopy(self._py_env)
                    except Exception:
                        # fallback: keep reference to live env (less safe)
                        self._start_env = self._py_env

                def get_random_action(self):
                    # produce a continuous-style action: (speed, heading)
                    if self._continuous:
                        speed = _np.random.random() * self._max_speed
                        heading = (_np.random.random() * 360.0) - 180.0
                        return (speed, heading)
                    else:
                        # discrete fallback
                        return int(_np.random.randint(0, 15))

                def evaluate_rollout(self, solutions, discount_factor=None, ignore_frames=0):
                    """
                    solutions: numpy array-like of shape (num_evals, rollout_length, ...) where ... is action representation
                    Return: 1D numpy array of scores for each solution
                    """
                    scores = []
                    # iterate candidate solutions
                    for sol in solutions:
                        # use a fresh copy of the start env
                        try:
                            sim_env = copy.deepcopy(self._start_env)
                        except Exception:
                            sim_env = copy.deepcopy(self._py_env)
                        total_reward = 0.0
                        discount = 1.0
                        # each element in sol is an action for the controlled agent
                        for action in sol:
                            # build action dict for all agents
                            action_dict = {}
                            # controlled agent action
                            action_dict[self._agent_id] = action
                            # other agents: keep them stationary / default
                            for tid in (self._teammate_ids + self._opponent_ids):
                                if tid != self._agent_id:
                                    action_dict[tid] = self._other_action
                            try:
                                # step the simulated env
                                obs, reward, terminated, truncated, info = sim_env.step(action_dict)
                            except Exception:
                                # older gym versions may return 4-tuple
                                try:
                                    obs, reward, done, info = sim_env.step(action_dict)
                                    terminated = done.get(self._agent_id, False) if isinstance(done, dict) else done
                                    truncated = False
                                except Exception:
                                    # can't step: give zero score
                                    terminated = True
                                    truncated = True
                                    reward = {self._agent_id: 0.0}
                            # accumulate reward for the agent
                            r = 0.0
                            if isinstance(reward, dict):
                                r = float(reward.get(self._agent_id, 0.0))
                            else:
                                # single-number reward
                                r = float(reward)
                            total_reward += discount * r
                            if discount_factor is not None:
                                discount *= discount_factor
                            # optional ignore frames: not used in this simple impl
                            if terminated or truncated:
                                break
                        scores.append(total_reward)
                    return _np.array(scores)

                def perform_action(self, action):
                    # direct perform a single action on the real env
                    action_dict = {self._agent_id: action}
                    for tid in (self._teammate_ids + self._opponent_ids):
                        if tid != self._agent_id:
                            action_dict[tid] = self._other_action
                    try:
                        self._py_env.step(action_dict)
                    except Exception:
                        pass

                def is_game_over(self):
                    # best-effort check: if the real env exposes done info/state, try it
                    try:
                        return getattr(self._py_env, "game_over", False)
                    except Exception:
                        return False

                def get_current_score(self):
                    # best-effort: return 0 (RHEA run() logging only)
                    return 0.0

            # create the wrapper and RHEA instance with reasonable defaults
            self._rhea_env_wrapper = SingleAgentRHEAEnv(
                env,
                self.id,
                self.teammate_ids,
                self.opponent_ids,
                continuous=self.continuous,
                max_speed=self.max_speed,
            )
            # Tune these hyper-parameters to your needs
            rollout_actions_length = 8
            mutation_probability = 0.2
            num_evals = 16
            self.rhea = RollingHorizonEvolutionaryAlgorithm(
                rollout_actions_length, self._rhea_env_wrapper, mutation_probability, num_evals
            )
        ######END of cpilot edit

    def set_mode(self, mode: str):
        """Sets difficulty mode."""
        if mode not in MODES:
            raise ValueError(f"mode {mode} not in set of valid modes: {MODES}")
        self.mode = mode

    def compute_action(self, obs, info):
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
        # Update the state based on this observation
        self.update_state(obs, info)

        if self.mode == "nothing":
            return self.action_from_vector(None, 0)

        # RHEA algorithm (only works for a single agent so far):
        if self.mode == "rhea":
            # use RHEA instance if available
            if self.rhea is None:
                # fallback to attacker if not initialized
                print("RHEA not initialized, falling back to failsafe")
                self.mode = "nothing"
                return self.compute_action(obs, info)
            # let the internal wrapper snapshot the current env state for rollouts
            try:
                return self.rhea.compute_action(obs, info)
            except Exception:
                # on any failure, fallback to base attacker
                print("Failure: on any failure, fallback to failsafe")
                self.mode = "nothing"
                return self.compute_action(obs, info)

        if self.mode == "easy":
            # Opp is close - needs to defend:
            if self.is_close_to_flag() and False in self.opp_team_tag:
                return self.base_defender.compute_action(obs, info)

            # Opp on defensive - needs to attack
            else:
                return self.base_attacker.compute_action(obs, info)

        else:
            # If I have the flag, just bring it back to base
            if self.has_flag:
                return self.base_attacker.compute_action(obs, info)

            elif self.opp_team_has_flag:
                return self.base_defender.compute_action(obs, info)

            # Opp is close - go on defensive
            elif self.is_close_to_flag() and (False in self.opp_team_tag):
                return self.base_defender.compute_action(obs, info)

            # Opp on defensive - needs to attack
            elif self.is_far_from_flag():
                return self.base_attacker.compute_action(obs, info)

            else:
                if self.mode == "hard":
                    return self.base_attacker.compute_action(obs, info)
                else:
                    return self.random_defense_action(self.opp_team_pos)

    def random_defense_action(self, enem_positions):
        """
        Randomly compute an action that steers the agent to it's own side of the field and sometimes
        towards its flag.
        """
        if np.random.random() < 0.25:
            # go to random point on segment between my flag and scrimmage line
            t = np.random.random()
            goal_vec = (1 - t) * self.my_flag_loc + t * self.midpoint_local

        else:
            near_enemy_dist = np.inf
            nearest_enemy = None
            for en in enem_positions:
                temp_enem_dist = en[0]
                if temp_enem_dist < near_enemy_dist:
                    near_enemy_dist = temp_enem_dist
                    nearest_enemy = en
            assert nearest_enemy is not None
            if np.random.random() < 0.5:
                goal_vec = rel_bearing_to_local_unit_rect(nearest_enemy[1])
            else:
                own_flag_dist = self.my_flag_distance
                if own_flag_dist > self.flag_keepout + 2.0:
                    goal_vec = rel_bearing_to_local_unit_rect(self.my_flag_bearing)
                else:
                    # want random point on segment between my flag and scrimmage line, but at least <defensiveness> meters from scrimmage line
                    t = np.random.random()
                    unit_vec = unit_vect_between_points(
                        self.midpoint_local, self.my_flag_loc
                    )
                    endpoint = (
                        self.midpoint_local
                        + min(
                            self.defensiveness,
                            dist(self.midpoint_local, self.my_flag_loc),
                        )
                        * unit_vec
                    )
                    goal_vec = (1 - t) * self.my_flag_loc + t * endpoint

        if not self.on_sides:
            goal_vec = goal_vec + get_avoid_vect(self.opp_team_pos, avoid_threshold=15)

        if self.mode == "hard":
            return self.action_from_vector(goal_vec, 1)
        else:
            return self.action_from_vector(goal_vec, 0.5)

    def get_team_density(self, friendly_positions, enemy_positions):
        """This function returns the center of mass and varience of all the agents in the team."""
        home_x = []
        home_y = []
        away_x = []
        away_y = []

        # TO DO: THIS NEEDS TO BE BY X and Y seperately
        for agp in friendly_positions:
            ag = dist_rel_bearing_to_local_rect(*agp)
            home_x.append(ag[0])
            home_y.append(ag[1])
        for agp in enemy_positions:
            ag = dist_rel_bearing_to_local_rect(*agp)
            away_x.append(ag[0])
            away_y.append(ag[1])

        home_mean = np.array([np.mean(home_x), np.mean(home_y)])
        home_std = np.mean(np.array([np.std(home_x), np.std(home_y)]))

        away_mean = np.array([np.mean(away_x), np.mean(away_y)])
        away_std = np.mean(np.array([np.std(away_x), np.std(away_y)]))

        return [home_mean, home_std], [away_mean, away_std]

    def is_close_to_flag(self, threshold=30):
        """
        Checks how close the opposing teams position is relative to the flag.

        Args:
            threshold: The threshold distance to use for comparison

        Returns
        -------
            True if the center of mass of the opposing team is within
            ``threshold`` units of the flag, False otherwise
        """
        if self.opp_team_density is None:
            raise RuntimeWarning(
                "Must call update_state() before trying to get an action."
            )

        dist_to_flag = dist(self.opp_team_density[0], np.array(self.my_flag_loc))
        return dist_to_flag < threshold

    def is_far_from_flag(self, threshold=50):
        """
        Checks how far the opposing teams position is relative to the flag.

        Args:
            threshold: The threshold distance to use for comparison

        Returns
        -------
            True if the center of mass of the opposing team is further than
            ``threshold`` units of the flag, False otherwise
        """

        if self.opp_team_density is None:
            raise RuntimeWarning(
                "Must call update_state() before trying to get an action."
            )

        dist_to_flag = dist(self.opp_team_density[0], np.array(self.my_flag_loc))
        return dist_to_flag > threshold

    def update_state(self, obs, info: dict[str, dict]) -> None:
        """
        Method to convert the gym obs and info into data more relative to the
        agent.

        Note: all rectangular positions are in the ego agent's local coordinate frame.
        Note: all bearings are relative, measured in degrees clockwise from the ego agent's heading.

        Args:
            obs: observation from gym
            info: info from gym
        """

        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        self.on_sides = global_state[(self.id, "on_side")]
        self.has_flag = global_state[(self.id, "has_flag")]

        # Copy the polar positions of each agent, separated by team and get their tag status
        self.opp_team_pos = []
        self.my_team_pos = []
        self.opp_team_tag = []
        self.opp_team_has_flag = False
        for id in self.teammate_ids:
            if id != self.id:
                distance = dist(
                    global_state[(self.id, "pos")], global_state[(id, "pos")]
                )
                bearing = angle180(
                    global_rect_to_abs_bearing(
                        global_state[(id, "pos")] - global_state[(self.id, "pos")]
                    )
                    - global_state[(self.id, "heading")]
                )
                self.my_team_pos.append(np.array((distance, bearing)))
        for id in self.opponent_ids:
            distance = dist(global_state[(self.id, "pos")], global_state[(id, "pos")])
            bearing = angle180(
                global_rect_to_abs_bearing(
                    global_state[(id, "pos")] - global_state[(self.id, "pos")]
                )
                - global_state[(self.id, "heading")]
            )
            self.opp_team_pos.append(np.array((distance, bearing)))
            self.opp_team_has_flag = (
                self.opp_team_has_flag or global_state[(id, "has_flag")]
            )
            self.opp_team_tag.append(global_state[(id, "is_tagged")])
        team_str = self.team.name.lower().split("_")[0]
        opp_str = "red" if team_str == "blue" else "blue"
        self.my_flag_distance = dist(
            global_state[(self.id, "pos")], global_state[team_str + "_flag_pos"]
        )
        self.my_flag_bearing = angle180(
            global_rect_to_abs_bearing(
                global_state[team_str + "_flag_pos"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.my_flag_loc = dist_rel_bearing_to_local_rect(
            self.my_flag_distance, self.my_flag_bearing
        )
        self.opp_flag_distance = dist(
            global_state[(self.id, "pos")], global_state[opp_str + "_flag_pos"]
        )
        self.opp_flag_bearing = angle180(
            global_rect_to_abs_bearing(
                global_state[opp_str + "_flag_pos"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.opp_flag_loc = dist_rel_bearing_to_local_rect(
            self.opp_flag_distance, self.opp_flag_bearing
        )

        self.my_team_density, self.opp_team_density = self.get_team_density(
            self.my_team_pos, self.opp_team_pos
        )

        self.midpoint_local = global_rect_to_local_rect(
            self.midpoint_global,
            global_state[(self.id, "pos")],
            global_state[(self.id, "heading")],
        )

    def action_from_vector(self, vector, desired_speed_normalized):
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
