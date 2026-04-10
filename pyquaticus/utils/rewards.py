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
    prev_num_oob = prev_state['agent_oob'][agents.index(agent_id)]
    num_oob = state['agent_oob'][agents.index(agent_id)]
    if num_oob > prev_num_oob:
        reward += -1.0
    # If close to out of bounds, start punishing
    position = np.array(state['agent_position'][agents.index(agent_id)])
    if position[0] > 155.0 or position[1] > 75.0 or position[0] < 5.0 or position[1] < 5.0:
        reward += -0.5
    for t in [0,1]:
        prev_num_grabs = prev_state['grabs'][t]
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 0.25 if t == team else -0.25

        prev_num_caps = prev_state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 1.0 if t == team else -1.0

    return reward

### Added Custom Reward Functions Here ###

def caps_and_tags(
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
    prev_num_oob = prev_state['agent_oob'][agents.index(agent_id)]
    num_oob = state['agent_oob'][agents.index(agent_id)]
    if num_oob > prev_num_oob:
        reward += -10.0
    # If close to out of bounds, start punishing
    position = np.array(state['agent_position'][agents.index(agent_id)])
    if position[0] > 155.0 or position[1] > 75.0 or position[0] < 5.0 or position[1] < 5.0:
        reward += -5.0
    for t in [0,1]:
        prev_num_grabs = prev_state['grabs'][t]
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 5.0 if t == team else -5.0

        prev_num_caps = prev_state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 10.0 if t == team else -10.0

    # awards points for tagging opponents and being tagged
    prev_num_tags = prev_state['tags'][team.value]
    num_tags = state['tags'][team.value]
    if num_tags > prev_num_tags:
        reward += 10.0 if t == team else -10.0

    return reward

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
    """NOTE: This reward is outdated, better to use single_aggressive_rew or a variation."""
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
    if position[0] > 160.0 or position[1] > 80.0 or position[0] < 0.0 or position[1] < 0.0: #my own implementation also doesn't work. Wtf?<-was likely just wrong reward selected...
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

#agressive_rew but as 3-agent adaptation (needs to compute all three at once)
def triple_aggressive_rew(
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
    #idx = agents.index(agent_id)
    idx1 = 0#'agent0'
    idx2 = 1#'agent1'
    idx3 = 2#'agent2'
    position1 = np.array(state['agent_position'][idx1])
    position2 = np.array(state['agent_position'][idx2])
    position3 = np.array(state['agent_position'][idx3])
    prev_position1 = np.array(prev_state['agent_position'][idx1])
    prev_position2 = np.array(prev_state['agent_position'][idx2])
    prev_position3 = np.array(prev_state['agent_position'][idx3])

    # If tagged, return minus one #exact number got adjusted
    if state['agent_is_tagged'][idx1]:
        reward += -5.0 
    if state['agent_is_tagged'][idx2]:
        reward += -5.0 
    if state['agent_is_tagged'][idx3]:
        reward += -5.0 

    # If out of bounds, return minus one
    if position1[0] > 160.0 or position1[1] > 80.0 or position1[0] < 0.0 or position1[1] < 0.0:
        reward += -10.0
    if position2[0] > 160.0 or position2[1] > 80.0 or position2[0] < 0.0 or position2[1] < 0.0:
        reward += -10.0
    if position3[0] > 160.0 or position3[1] > 80.0 or position3[0] < 0.0 or position3[1] < 0.0:
        reward += -10.0

    # Determine flag homes
    flag_homes = np.array(state['flag_home'])
    if isinstance(team, Team):
        t = team.value 
    else:
        t = 0 if str(team).lower() == 'blue_team' else 1
    team_home = flag_homes[t]
    opp_home = flag_homes[(t + 1) % 2]
    # Determine which flag to aim for
    has_flag1 = bool(state['agent_has_flag'][idx1])
    has_flag2 = bool(state['agent_has_flag'][idx2])
    has_flag3 = bool(state['agent_has_flag'][idx3])
    # Go to the enemy flag, if grabbed flag then go to own base.
    target_flag_pos1 = team_home if has_flag1 else opp_home
    target_flag_pos2 = team_home if has_flag2 else opp_home
    target_flag_pos3 = team_home if has_flag3 else opp_home

    # Reward movement toward target
    # Agent 1
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1
    # Agent 2
    prev_diff2 = prev_position2 - target_flag_pos2
    curr_diff2 = position2 - target_flag_pos2
    # Agent 3
    prev_diff3 = prev_position3 - target_flag_pos3
    curr_diff3 = position3 - target_flag_pos3
    rewardable_movement = numpy.sqrt(np.sum(prev_diff1**2)) - numpy.sqrt(np.sum(curr_diff1**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        reward += rewardable_movement / max_speeds[0]
    rewardable_movement = numpy.sqrt(np.sum(prev_diff2**2)) - numpy.sqrt(np.sum(curr_diff2**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        reward += rewardable_movement / max_speeds[0]
    rewardable_movement = numpy.sqrt(np.sum(prev_diff3**2)) - numpy.sqrt(np.sum(curr_diff3**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        reward += rewardable_movement / max_speeds[0]

    # Capture and grab bonuses
    num_grabs = state['grabs'][t]
    num_caps = state['captures'][t]
    prev_num_grabs = prev_state['grabs'][t]
    prev_num_caps = prev_state['captures'][t]
    reward += 30 * (num_caps - prev_num_caps) + 30 * (num_grabs - prev_num_grabs)
    return reward

# Three agent implementation of caps_and_grabs
def triple_caps_and_grabs(
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
    agent_id1 = 0
    agent_id2 = 1
    agent_id3 = 2
    reward = 0.0
    prev_num_oob1 = prev_state['agent_oob'][agent_id1]
    num_oob1 = state['agent_oob'][agent_id1]
    prev_num_oob2 = prev_state['agent_oob'][agent_id2]
    num_oob2 = state['agent_oob'][agent_id2]
    prev_num_oob3 = prev_state['agent_oob'][agent_id3]
    num_oob3 = state['agent_oob'][agent_id3]
    if num_oob1 > prev_num_oob1:
        reward += -5000000.0
    if num_oob2 > prev_num_oob2:
        reward += -5000000.0
    if num_oob3 > prev_num_oob3:
        reward += -5000000.0
    for agent_id in [agent_id1, agent_id2, agent_id3]:
        position1 = np.array(state['agent_position'][agent_id])
        # Slowly start punishing if close to out of bounds: (hardcoded for the left side currently)
        if position1[0] > 155.0 or position1[1] > 75.0 or position1[0] < 5.0 or position1[1] < 5.0:
            reward += -5000.0
    for t in [0,1]:
        prev_num_grabs = prev_state['grabs'][t]
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 250.0 if t == team else -250.0

        prev_num_caps = prev_state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 1000.0 if t == team else -1000.0

    return reward

def defensive_rew(
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

    # Current and previous position of this agent
    position = np.array(state["agent_position"][idx])
    prev_position = np.array(prev_state["agent_position"][idx])

    # Punish being tagged or out of bounds
    if state["agent_is_tagged"][idx]:
        return -5.0  
    # oob check:
    if (
        position[0] < 0 or position[0] > env_size[0] or
        position[1] < 0 or position[1] > env_size[1]
    ):
        return -10.0
    # If close to out of bounds, start punishing
    if position[0] > (env_size[0] - 5.) or position[1] > (env_size[1] - 5.) or position[0] < 5.0 or position[1] < 5.0:
        reward += -5.0 #TODO test if correct

    # Note team index
    t = team.value  #0 (blue team) or 1 (red team)

    enemy_team = (t + 1) % 2

    # Flag position of own team (defensive target)
    own_flag_pos = np.array(state["flag_home"][t])

    # Enemy agent indices
    #enemy_agent_inds = agents.index(ag     #agent_inds_of_team[enemy_team]
    enemy_team_enum = Team(enemy_team)
    enemy_agent_ids = agent_inds_of_team[enemy_team_enum]


    # Identify the enemy that is the biggest threat
    best_threat = None
    best_threat_dist = float("inf")

    for e_idx in enemy_agent_ids:
        enemy_pos = np.array(state["agent_position"][e_idx])
        has_flag = state["agent_has_flag"][e_idx]

        if has_flag:
            # Highest priority threat
            dist = 0.0
            #dist = np.linalg.norm(enemy_pos - own_flag_pos) * 0.01
        else:
            # Standard threat: distance to your flag
            dist = np.linalg.norm(enemy_pos - own_flag_pos)

        if dist < best_threat_dist:
            best_threat_dist = dist
            best_threat = e_idx

    # No enemy agents possible?
    if best_threat is None:
        print("Error. No best threat found.")
        return 0.0

    # Compute defensive movement reward
    threat_pos = np.array(state["agent_position"][best_threat])

    prev_dist_to_threat = np.linalg.norm(prev_position - threat_pos)
    curr_dist_to_threat = np.linalg.norm(position - threat_pos)

    print("Best Enemy is: ",best_threat, "location: ",threat_pos, "distance: ",curr_dist_to_threat, "prev_dist: ",prev_dist_to_threat)

    # Reward reducing distance to the threat
    distance_improvement = prev_dist_to_threat - curr_dist_to_threat

    if distance_improvement > 0:
        # Normalize by max speed
        reward += (min(1.0, distance_improvement / max_speeds[0]) * 5.0)
    else:
        reward += distance_improvement   #penalty for moving away

    # Bonus for tagging the threat
    if (
        state["agent_is_tagged"][best_threat] and 
        not prev_state["agent_is_tagged"][best_threat]
    ):
        reward += 5.0

    print("reward=",reward)
    return reward

#agressive_rew but as 2-agent adaptation (needs to compute all two at once) TODO or do we want one-agent reward?
def double_aggressive_rew(
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
    #idx = agents.index(agent_id)
    idx1 = 0#'agent0'
    idx2 = 1#'agent1'
    #idx3 = 2#'agent2'
    position1 = np.array(state['agent_position'][idx1])
    position2 = np.array(state['agent_position'][idx2])
    #position3 = np.array(state['agent_position'][idx3])
    prev_position1 = np.array(prev_state['agent_position'][idx1])
    prev_position2 = np.array(prev_state['agent_position'][idx2])
    #prev_position3 = np.array(prev_state['agent_position'][idx3])

    # If tagged, return minus one #exact number got adjusted
    if state['agent_is_tagged'][idx1]:
        reward += -5.0 
    if state['agent_is_tagged'][idx2]:
        reward += -5.0 
    #if state['agent_is_tagged'][idx3]:
    #    reward += -5.0 

    # If out of bounds, return minus one
    if position1[0] > 160.0 or position1[1] > 80.0 or position1[0] < 0.0 or position1[1] < 0.0:
        reward += -10.0
    if position2[0] > 160.0 or position2[1] > 80.0 or position2[0] < 0.0 or position2[1] < 0.0:
        reward += -10.0
    #if position3[0] > 160.0 or position3[1] > 80.0 or position3[0] < 0.0 or position3[1] < 0.0:
    #    reward += -10.0
    # If close to out of bounds, start punishing
    for position in [position1, position2]:#, position3]:
        if position[0] > 155.0 or position[1] > 75.0 or position[0] < 5.0 or position[1] < 5.0:
            reward += -5.0

    # Determine flag homes
    flag_homes = np.array(state['flag_home'])
    if isinstance(team, Team):
        t = team.value 
    else:
        t = 0 if str(team).lower() == 'blue_team' else 1
    team_home = flag_homes[t]
    opp_home = flag_homes[(t + 1) % 2]
    # Determine which flag to aim for
    has_flag1 = bool(state['agent_has_flag'][idx1])
    has_flag2 = bool(state['agent_has_flag'][idx2])
    #has_flag3 = bool(state['agent_has_flag'][idx3])
    # Go to the enemy flag, if grabbed flag then go to own base.
    target_flag_pos1 = team_home if has_flag1 else opp_home
    target_flag_pos2 = team_home if has_flag2 else opp_home
    #target_flag_pos3 = team_home if has_flag3 else opp_home

    # Reward movement toward target
    # Agent 1
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1
    # Agent 2
    prev_diff2 = prev_position2 - target_flag_pos2
    curr_diff2 = position2 - target_flag_pos2
    # Agent 3
    #prev_diff3 = prev_position3 - target_flag_pos3
    #curr_diff3 = position3 - target_flag_pos3
    rewardable_movement = numpy.sqrt(np.sum(prev_diff1**2)) - numpy.sqrt(np.sum(curr_diff1**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        reward += rewardable_movement / max_speeds[0]
    rewardable_movement = numpy.sqrt(np.sum(prev_diff2**2)) - numpy.sqrt(np.sum(curr_diff2**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    #else:
    #    reward += rewardable_movement / max_speeds[0]
    #rewardable_movement = numpy.sqrt(np.sum(prev_diff3**2)) - numpy.sqrt(np.sum(curr_diff3**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        reward += rewardable_movement / max_speeds[0]

    # Capture and grab bonuses
    num_grabs = state['grabs'][t]
    num_caps = state['captures'][t]
    prev_num_grabs = prev_state['grabs'][t]
    prev_num_caps = prev_state['captures'][t]
    reward += 30 * (num_caps - prev_num_caps) + 30 * (num_grabs - prev_num_grabs)
    return reward

# One agent version of aggressive_rew, which only computes the reward for one agent
def single_aggressive_rew(
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
    """Modified from double_aggressive_rew to be calculated for a single agent (even if it works as part of a larger team).
    Contains a positional reward (move closer to flag) and rewards for grabbing/capturing the flag as well as negative 
    rewards for getting tagged and moving out of bounds."""
    #previousreward = state.get('previous_reward', 0.0)  # Get previous reward if it exists, otherwise default to 0.0 TODO
    reward = 0.0
    idx1 = agents.index(agent_id)
    #idx1 = agent_id#0#'agent0'
    #print("idx: ",idx1)
    #idx2 = 1#'agent1'
    #idx3 = 2#'agent2'
    position1 = np.array(state['agent_position'][idx1])
    #position2 = np.array(state['agent_position'][idx2])
    #position3 = np.array(state['agent_position'][idx3])
    prev_position1 = np.array(prev_state['agent_position'][idx1])
    #prev_position2 = np.array(prev_state['agent_position'][idx2])
    #prev_position3 = np.array(prev_state['agent_position'][idx3])

    # If tagged, return minus one #exact number got adjusted
    if state['agent_is_tagged'][idx1]:
        reward += -5.0 
    #if state['agent_is_tagged'][idx2]:
    #    reward += -5.0 
    #if state['agent_is_tagged'][idx3]:
    #    reward += -5.0 

    # If out of bounds, return minus one
    if position1[0] > 160.0 or position1[1] > 80.0 or position1[0] < 0.0 or position1[1] < 0.0:
        reward += -10.0
    #if position2[0] > 160.0 or position2[1] > 80.0 or position2[0] < 0.0 or position2[1] < 0.0:
    #    reward += -10.0
    #if position3[0] > 160.0 or position3[1] > 80.0 or position3[0] < 0.0 or position3[1] < 0.0:
    #    reward += -10.0
    # If close to out of bounds, start punishing
    if position1[0] > 155.0 or position1[1] > 75.0 or position1[0] < 5.0 or position1[1] < 5.0:
        reward += -5.0

    # Determine flag homes
    flag_homes = np.array(state['flag_home'])
    if isinstance(team, Team):
        t = team.value 
    else:
        t = 0 if str(team).lower() == 'blue_team' else 1
    team_home = flag_homes[t]
    opp_home = flag_homes[(t + 1) % 2]
    # Determine which flag to aim for
    has_flag1 = bool(state['agent_has_flag'][idx1])
    #has_flag2 = bool(state['agent_has_flag'][idx2]) TODO maybe this should be kept for the one agent version as well?
    #has_flag3 = bool(state['agent_has_flag'][idx3])
    # Go to the enemy flag, if grabbed flag then go to own base.
    target_flag_pos1 = team_home if has_flag1 else opp_home
    #target_flag_pos2 = team_home if has_flag2 else opp_home
    #target_flag_pos3 = team_home if has_flag3 else opp_home

    # Reward movement toward target
    # Agent 1
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1
    # Agent 2
    #prev_diff2 = prev_position2 - target_flag_pos2
    #curr_diff2 = position2 - target_flag_pos2
    # Agent 3
    #prev_diff3 = prev_position3 - target_flag_pos3
    #curr_diff3 = position3 - target_flag_pos3
    rewardable_movement = numpy.sqrt(np.sum(prev_diff1**2)) - numpy.sqrt(np.sum(curr_diff1**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        reward += rewardable_movement / max_speeds[0]

    # Capture and grab bonuses
    num_grabs = state['grabs'][t]
    num_caps = state['captures'][t]
    prev_num_grabs = prev_state['grabs'][t]
    prev_num_caps = prev_state['captures'][t]
    reward += 30 * (num_caps - prev_num_caps) + 30 * (num_grabs - prev_num_grabs)
    return reward



    
def aggr_rew_alt(
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
    """copied from single_aggressive_rew...
    
    single_aggressive_rew but with higher positive rewards, since final 
    rewardsum often turned out to be high negative values or very small 
    positive values, to test if this results in agents finding non-local 
    optimums."""
    reward = 0.0
    idx1 = agents.index(agent_id)
    position1 = np.array(state['agent_position'][idx1])
    prev_position1 = np.array(prev_state['agent_position'][idx1])

    # If tagged, return minus one #exact number got adjusted
    if state['agent_is_tagged'][idx1]:
        reward += -5.0 

    # If out of bounds, return minus one
    if position1[0] > 160.0 or position1[1] > 80.0 or position1[0] < 0.0 or position1[1] < 0.0:
        reward += -10.0
    # If close to out of bounds, start punishing
    if position1[0] > 155.0 or position1[1] > 75.0 or position1[0] < 5.0 or position1[1] < 5.0:
        reward += -5.0
    if position1[0] > 150.0 or position1[1] > 70.0 or position1[0] < 10.0 or position1[1] < 10.0:
        reward += -1.0

    # Determine flag homes
    flag_homes = np.array(state['flag_home'])
    if isinstance(team, Team):
        t = team.value 
    else:
        t = 0 if str(team).lower() == 'blue_team' else 1
    team_home = flag_homes[t]
    opp_home = flag_homes[(t + 1) % 2]
    # Determine which flag to aim for
    has_flag1 = bool(state['agent_has_flag'][idx1])
    # Go to the enemy flag, if grabbed flag then go to own base.
    target_flag_pos1 = team_home if has_flag1 else opp_home

    # Reward movement toward target
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1

    rewardable_movement = numpy.sqrt(np.sum(prev_diff1**2)) - numpy.sqrt(np.sum(curr_diff1**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        if rewardable_movement > 0.:
            rewardable_movement = rewardable_movement * 5
        reward += rewardable_movement / max_speeds[0]

    # Capture and grab bonuses
    num_grabs = state['grabs'][t]
    num_caps = state['captures'][t]
    prev_num_grabs = prev_state['grabs'][t]
    prev_num_caps = prev_state['captures'][t]
    reward += 30 * (num_caps - prev_num_caps) + 30 * (num_grabs - prev_num_grabs)
    return reward

def aggressive_tags_24(
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
    """Modified from double_aggressive_rew to be calculated for a single agent (even if it works as part of a larger team).
    Contains a positional reward (move closer to flag) and rewards for grabbing/capturing the flag as well as negative 
    rewards for getting tagged and moving out of bounds."""
    #previousreward = state.get('previous_reward', 0.0)  # Get previous reward if it exists, otherwise default to 0.0 TODO
    reward = 0.0
    idx1 = agents.index(agent_id)
    #idx1 = agent_id#0#'agent0'
    #print("idx: ",idx1)
    #idx2 = 1#'agent1'
    #idx3 = 2#'agent2'
    position1 = np.array(state['agent_position'][idx1])
    #position2 = np.array(state['agent_position'][idx2])
    #position3 = np.array(state['agent_position'][idx3])
    prev_position1 = np.array(prev_state['agent_position'][idx1])
    #prev_position2 = np.array(prev_state['agent_position'][idx2])
    #prev_position3 = np.array(prev_state['agent_position'][idx3])

    # If tagged, return minus one #exact number got adjusted
    if state['agent_is_tagged'][idx1]:
        reward += -5.0 
    #if state['agent_is_tagged'][idx2]:
    #    reward += -5.0 
    #if state['agent_is_tagged'][idx3]:
    #    reward += -5.0 

    # If out of bounds, return minus one
    if position1[0] > 160.0 or position1[1] > 80.0 or position1[0] < 0.0 or position1[1] < 0.0:
        reward += -10.0
    #if position2[0] > 160.0 or position2[1] > 80.0 or position2[0] < 0.0 or position2[1] < 0.0:
    #    reward += -10.0
    #if position3[0] > 160.0 or position3[1] > 80.0 or position3[0] < 0.0 or position3[1] < 0.0:
    #    reward += -10.0
    # If close to out of bounds, start punishing
    if position1[0] > 155.0 or position1[1] > 75.0 or position1[0] < 5.0 or position1[1] < 5.0:
        reward += -5.0

    # Determine flag homes
    flag_homes = np.array(state['flag_home'])
    if isinstance(team, Team):
        t = team.value 
    else:
        t = 0 if str(team).lower() == 'blue_team' else 1
    team_home = flag_homes[t]
    opp_home = flag_homes[(t + 1) % 2]
    # Determine which flag to aim for
    has_flag1 = bool(state['agent_has_flag'][idx1])
    #has_flag2 = bool(state['agent_has_flag'][idx2]) TODO maybe this should be kept for the one agent version as well?
    #has_flag3 = bool(state['agent_has_flag'][idx3])
    # Go to the enemy flag, if grabbed flag then go to own base.
    target_flag_pos1 = team_home if has_flag1 else opp_home
    #target_flag_pos2 = team_home if has_flag2 else opp_home
    #target_flag_pos3 = team_home if has_flag3 else opp_home

    # Reward movement toward target
    # Agent 1
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1
    # Agent 2
    #prev_diff2 = prev_position2 - target_flag_pos2
    #curr_diff2 = position2 - target_flag_pos2
    # Agent 3
    #prev_diff3 = prev_position3 - target_flag_pos3
    #curr_diff3 = position3 - target_flag_pos3
    rewardable_movement = numpy.sqrt(np.sum(prev_diff1**2)) - numpy.sqrt(np.sum(curr_diff1**2))
    if rewardable_movement > max_speeds[0]:
        reward += 1.0
    else:
        reward += rewardable_movement / max_speeds[0]

    # Capture and grab bonuses
    num_grabs = state['grabs'][t]
    num_caps = state['captures'][t]
    prev_num_grabs = prev_state['grabs'][t]
    prev_num_caps = prev_state['captures'][t]
    reward += 30 * (num_caps - prev_num_caps) + 30 * (num_grabs - prev_num_grabs)

    # Points for tagging opponents and being tagged
    prev_num_tags = prev_state['tags'][team.value]
    num_tags = state['tags'][team.value]
    if num_tags > prev_num_tags:
        reward += 10.0 if t == team else -10.0
    return reward
