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

# Modified as part of the Masters Project Oct 2025

"""
#Configureable Rewards
    # -- NOTE --
    #   All headings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    #
    # This can be converted the standard heading format that is counterclockwise
    # by using the heading_angle_conversion(deg) function found in utils.py
    #
    #
    ## Each custom reward function should have the following arguments ##
    Args:
        agent_id (int): ID of the agent we are computing the reward for
        team (Team): team of the agent we are computing the reward for
        agents (list): list of agent ID's (this is used to map agent_id's to agent indices and viceversa)
        agent_inds_of_team (dict): mapping from team to agent indices of that team
        state (dict):
            'agent_position' (array): list of agent positions (indexed in the order of agents list)

                        Ex. Usage: Get agent's current position
                        agent_id = 'agent_1'
                        position = state['agent_position'][agents.index(agent_id)]

            'prev_agent_position' (array): list of agent positions (indexed in the order of agents list) at the previous timestep

                        Ex. Usage: Get agent's previous position
                        agent_id = 'agent_1'
                        prev_position = state['prev_agent_position'][agents.index(agent_id)]

            'agent_speed' (array): list of agent speeds (indexed in the order of agents list)

                        Ex. Usage: Get agent's speed
                        agent_id = 'agent_1'
                        speed = state

            'agent_heading' (array): list of agent headings (indexed in the order of agents list)

                        Ex. Usage: Get agent's heading
                        agent_id = 'agent_1'
                        heading = state['agent_heading'][agents.index(agent_id)]

            'agent_on_sides' (array): list of booleans (indexed in the order of agents list) where True means the
                                      agent is on its own side, and False means the agent is not on its own side

                        Ex. Usage: Check if agent is on its own side
                        agent_id = 'agent_1'
                        on_own_side = state['agent_on_sides'][agents.index(agent_id)]

            'agent_oob' (array): list of booleans (indexed in the order of agents list) where True means the
                                 agent is out-of-bounds (OOB), and False means the agent is not out-of-bounds
                        
                        Ex. Usage: Check if agent is out-of-bounds
                        agent_id = 'agent_1'
                        num_oob = state['agent_oob'][agents.index(agent_id)]
            
            'agent_has_flag' (array): list of booleans (indexed in the order of agents list) where True means the
                                     agent has a flag, and False means the agent does not have a flag

                        Ex. Usage: Check if agent has a flag
                        agent_id = 'agent_1'
                        has_flag = state['agent_has_flag'][agents.index(agent_id)]

            'agent_is_tagged' (array): list of booleans (indexed in the order of agents list) where True means
                                       the agent is tagged, and False means the agent is not tagged

                        Ex. Usage: Check if agent is tagged
                        agent_id = 'agent_1'
                        is_tagged = state['agent_is_tagged'][agents.index(agent_id)]

            'agent_made_tag' (array): list (indexed in the order of agents list) where the value at an entry is the index of a different
                                     agent which the agent at the given index has tagged at the current timestep, otherwise None

                        Ex. Usage: Check if agent has tagged an agent
                        agent_id = 'agent_1'
                        tagged_opponent_idx = state['agent_made_tag'][agents.index(agent_id)]

            'agent_tagging_cooldown' (array): current agent tagging cooldowns (indexed in the order of agents list)
                        Note: agent is able to tag when this value is equal to tagging_cooldown
    
                        Ex. Usage: Get agent's current tagging cooldown
                        agent_id = 'agent_1'
                        cooldown = self.state['agent_tagging_cooldown'][agents.index(agent_id)]

            'dist_bearing_to_obstacles' (dict): For each agent in game list out distances and bearings
                                                to all obstacles in game in order of obstacles list

            'flag_home' (array): list of flag homes (indexed by team number)

            'flag_position' (array): list of flag homes (indexed by team number)

            'flag_taken' (array): list of booleans (indexed by team number) where True means the team's flag
                                  is taken (picked up by an opponent), and False means the flag is not taken 

            'team_has_flag' (array): list of booleans (indexed by team number) where True means an agent of the
                                     team has a flag, and False means that no agents are in possesion of a flag

            'captures' (array): list of total captures made by each team (indexed by team number)

            'tags' (array): list of total tags made by each team (indexed by team number)

            'grabs' (array): list of total flag grabs made by each team (indexed by team number)

            'agent_collisions' (array): list of total agent collisions  for each agent (indexed in the order of agents list)

            'agent_dynamics' (array): list of dictionaries containing agent-specific dynamics information (state attribute of a dynamics class - see dynamics.py)

            ######################################################################################
            ##### The following keys will exist in the state dictionary if lidar_obs is True #####
                'lidar_labels' (dict):

                'lidar_labels' (dict):

                'lidar_labels' (dict):
            ######################################################################################
            
            'obs_hist_buffer' (dict): Observation history buffer where the keys are agent_id's and values are the agents' observations

            'global_state_hist_buffer' (array): Global state history buffer

        prev_state (dict): Contains the state information from the previous step

        env_size (array): field dimensions [horizontal, vertical]

        agent_radii (array): list of agent radii (indexed in the order of agents list)

        catch_radius (float): tag and flag grab radius

        scrimmage_coords (array): endpoints [x,y] of the scrimmage line

        max_speeds (list): list of agent max speeds (indexed in the order of agents list)

        tagging_cooldown (float): tagging cooldown time
"""

import math
import numpy

from pyquaticus.structs import Team
from pyquaticus.utils.utils import *

### Example Reward Funtion ###
def example_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    return 0.0

def caps_and_grabs(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    reward = 0.0
    prev_num_oob = prev_state['agent_oob'][agents.index(agent_id)]#remove prev_
    num_oob = state['agent_oob'][agents.index(agent_id)]
    if num_oob > prev_num_oob:
        reward += -1.0
    for t in [0,1]:
        prev_num_grabs = prev_state['grabs'][t]#remove prev_
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 0.25 if t == team else -0.25

        prev_num_caps = prev_state['captures'][t]#remove prev_
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 1.0 if t == team else -1.0

    return reward

### Added Custom Reward Functions Here ###

def test_reward_func(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    reward = 0.0
    idx = agents.index(agent_id)
    position = state['agent_position'][idx]#numpy.array(state['agent_position'][idx])
    # # End of Trying capsngrabs

    print("reward function is outdated, use aggressive_rew or other")
    return reward
# End of test_reward_func()

# New attempt to create above function, by elitism copy pasting we create a cleaner and better working solution (hopefully):
def aggressive_rew(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    reward = 0.0
    idx = agents.index(agent_id)
    #print("idx: ",idx)
    position = np.array(state['agent_position'][idx])
    #print("position=",position) #<-Position appears to be incorrect. Perhaps wrong agent tracked?
    prev_position = np.array(prev_state['agent_position'][idx])
    # If tagged, return minus one
    # print("taggedness: ", state['agent_is_tagged'])
    """
    The tagged-counterreward would work better if it worked correctly in pyquaticus. (?)
    It appears in testing that the state['tagged'] is sometimes True when the agent 
    is NOT tagged and is False when the agent is not currently in the first frame of 
    being tagged.
    Similarly out of bounds is not always correctly shown in state[oob]. Sometimes it
    is the agents fault though for not being able to find a path that doesn't lead to
    the minus 10 counterreward... <-oob appears to be a count in the example function, 
    so this might be an outdated stat? Now it seems to be True/False for each agent...
    UPDATE: This was likely due to asynchron position. Since the env is not copyable 
    and renderable at the same time the copy has to be updated at every step. An own 
    step is not sufficient for this, deepcopying the state is not sufficient, some
    additional functions have to be called to update some variables using the state...
    """
    if state['agent_is_tagged'][idx]:
        # print("agent tagged, return -1")
        # print("reward is: ",-10.0)
        return -5.0 #only little counterreward because tagged appears to be True more often than it should?
    # If out of bounds, return minus one
    # if state['agent_oob'][idx]: #not trusting pyquaticus info. Computing oob myself...
    if position[0] > 160.0 or position[1] > 80.0 or position[0] < 0.0 or position[1] < 0.0: #my own implementation also doesn't work. Wtf?
    #     # print("agent_oob, return -1")
    #     # print("reward is: ",-1.0)
        return -10.0
    # prev_num_oob = prev_state['agent_oob'][idx]#remove prev_
    # num_oob = state['agent_oob'][idx]
    # print("agent_oob: ",state['agent_oob'])
    # if num_oob > prev_num_oob:
    #     reward += -10.0

    # Determine flag homes
    flag_homes = np.array(state['flag_home'])
    if isinstance(team, Team):
        t = team.value  # correct team index
    else:
        t = 0 if str(team).lower() == 'blue_team' else 1

    team_home = flag_homes[t]
    opp_home = flag_homes[(t + 1) % 2]
    #print("team_home: ",team_home,", t=",t,", opp_home: ",opp_home)

    # Distance metrics i dont want the total flag distance anymore
    #total_dist_between_flags = np.linalg.norm(team_home - opp_home)
    #total_dist_between_flags = max(total_dist_between_flags, np.hypot(*env_size))

    # Determine which flag to aim for
    has_flag = bool(state['agent_has_flag'][idx])
    #print("has_flag: ",has_flag)
    # Go to the enemy flag, if grabbed flag then go to own base.
    target_flag_pos = team_home if has_flag else opp_home

    # Reward movement toward target
    prev_diff = prev_position - target_flag_pos#np.linalg.norm(prev_position - target_flag_pos)
    curr_diff = position - target_flag_pos#np.linalg.norm(position - target_flag_pos)
    #reward += (prev_dist - curr_dist) / total_dist_between_flags  # positive if moving closer
    rewardable_movement = numpy.sqrt(np.sum(prev_diff**2)) - numpy.sqrt(np.sum(curr_diff**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
        # print("reward + 1.0")
    else:
        reward += rewardable_movement / max_speeds[0]
        # print("reward + ",rewardable_movement / max_speeds[0])
    # this is a cursed line (highlighted by vscodums python syntax highlighter, but for no reason)
    # Reward keeping distance from enemies when on enemy half
    # on_enemy_half = not bool(state['agent_on_sides'][idx])
    # if on_enemy_half:
    #     opp_positions = [ 
    #         np.array(state['agent_position'][i])
    #         for te, inds in agent_inds_of_team.items()
    #         if te != team for i in inds
    #     ]
    #     if opp_positions:
    #         min_dist = min(np.linalg.norm(position - p) for p in opp_positions)
    #         reward += 0.3 * min(1.0, min_dist / total_dist_between_flags)

    # Capture and grab bonuses
    num_grabs = state['grabs'][t]
    num_caps = state['captures'][t]
    prev_num_grabs = prev_state['grabs'][t]
    prev_num_caps = prev_state['captures'][t]
    reward += 5 * (num_caps - prev_num_caps) + 5 * (num_grabs - prev_num_grabs)
    # if not num_caps == prev_num_caps or not num_grabs == prev_num_grabs:
    #     print("reward + ",(5 * (num_caps - prev_num_caps) + 5 * (num_grabs - prev_num_grabs)))
    # print("reward is: ",reward)

    return reward