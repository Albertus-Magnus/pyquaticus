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

    #Check if agents lost flag
    prev_has_flag = prev_state['agent_has_flag'][agents.index(agent_id)]
    has_flag = state['agent_has_flag'][agents.index(agent_id)]
    #Agent lost flag
    if (prev_has_flag > has_flag): 
        reward += -0.25
    
    #Grabs and captures are of shape [team_0 (BLUE), team_1 (RED)] the value at the index 0 corresponds to the number of grabs
    for t in range(len(state['grabs'])):
        prev_num_grabs = prev_state['grabs'][t]
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 0.25 if t == int(team) else -0.25

        prev_num_caps = prev_state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 1.0 if t == int(team) else -1.0

    return reward

### Add Custom Reward Functions Here ###


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
    """Reward for captures, grabs, tags. Negative 
    reward for opponent captures, grabs, tags and 
    for oob."""
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



# One agent aggressive reward
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
    #target_flag_pos2 = team_home if has_flag2 else opp_home #TODO if team_home movement should take into account either both bases (square sum of distance?? or simple sum?) or the 'best' base, since in these rules there are two

    # Reward movement toward target
    # Agent 1
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1
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
    reward += 300 * (num_caps - prev_num_caps) + 300 * (num_grabs - prev_num_grabs)
    return reward

# One agent aggressive reward -- 2026-env version
def single_aggressive26(
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
    """Modified from single_aggressive_rew to be compatible with the 2026-comp environment (different home base to deliver flag to)
    [...]calculated for a single agent (even if it works as part of a larger team).
    Contains a positional reward (move closer to flag) and rewards for grabbing/capturing the flag as well as negative 
    rewards for getting tagged and moving out of bounds."""
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
    # Different home base to deliver flag to: (26env change)
    target_flag_pos1 = np.array([10., 70.]) if has_flag1 else opp_home

    # Reward movement toward target
    # Agent 1
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1
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
    reward += 300 * (num_caps - prev_num_caps) + 300 * (num_grabs - prev_num_grabs)
    return reward


# One agent aggressive reward
def aggressive_tags_26(
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
    """Modified from single_aggressive_rew to also reward tagging opponents.
    Adjusted to work well in the 2026 MCTF environment.
    """
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

    # Determine flag homes
    flag_homes = np.array(state['flag_home'])
    if isinstance(team, Team):
        t = team.value 
    else:
        t = 0 if str(team).lower() == 'blue_team' else 1
    team_home = np.array([5.0, 75.0])#.env_size[1] - 5.])    #flag_homes[t]                               #TODO Ugly quick fix, this needs to be adjusted to a better location to steer for. but currently this works reasonably well.
    opp_home = flag_homes[(t + 1) % 2]
    # Determine which flag to aim for
    has_flag1 = bool(state['agent_has_flag'][idx1])
    #has_flag2 = bool(state['agent_has_flag'][idx2]) TODO maybe this should be kept for the one agent version as well?
    #has_flag3 = bool(state['agent_has_flag'][idx3])

    # Go to the enemy flag, if grabbed flag then go to own base.
    target_flag_pos1 = team_home if has_flag1 else opp_home
    #target_flag_pos2 = team_home if has_flag2 else opp_home #TODO if team_home movement should take into account either both bases (square sum of distance?? or simple sum?) or the 'best' base, since in these rules there are two

    # Reward movement toward target
    # Agent 1
    prev_diff1 = prev_position1 - target_flag_pos1
    curr_diff1 = position1 - target_flag_pos1
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
    reward += 300 * (num_caps - prev_num_caps) + 300 * (num_grabs - prev_num_grabs)

    # awards points for tagging opponents (+) and being tagged (-)
    prev_num_tags = prev_state['tags'][team.value]
    num_tags = state['tags'][team.value]
    if num_tags > prev_num_tags:
        reward += 10.0 if t == team else -10.0

    return reward