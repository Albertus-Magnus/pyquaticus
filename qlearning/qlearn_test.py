from datetime import datetime
#import logging
import sys
#import os
#import os.path
#import pyquaticus
import numpy as np
from numpy.typing import NDArray
from pyquaticus import pyquaticus_v0
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv #2026 mctf environment
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_attack26 import BaseAttacker
# from pyquaticus.base_policies.multi_rhea_policy import MRHEA_Agent, MRHEA_Environment
# from pyquaticus.base_policies.rhealg_policy2 import RHEA_Agent, RHEA_Environment
# from pyquaticus.base_policies.ultra_def_policy import UltraDefender
from qtable import QlearnPolicy, QTable
from pyquaticus.utils.rewards import caps_and_grabs, single_aggressive_rew, caps_and_tags, aggressive_tags_26, single_aggressive26
from pyquaticus.mctf26_config import config_dict_std as mctf_config
#from multiprocessing import Pool, Value, Lock #i don't need any parallel processing (is qlearn even compatible?), i just need to run 10 scripts in different terminals...

"""
This was copied from magnus_test.py and modified to 
create a test of 2x qlearn policy agents versus 
2x base_combined agents during the Masters thesis (Feb 2026).
"""

# To run the training this is called as a function, with MODE and reward function as parameters.

def train_qlearn(
    #rewardcurve,
    #scores,
    #grabslist,
    s_table: NDArray,
    seed: int = 12345,
    # seed for "random" starts
    difficulty: str = "hard",
    # difficulty is the MODE of the example agents, can be "hard", "medium" or "easy"
    reward_choice: str = "adjustmepls", #maybe could be string, but cleaner so?
    render_mode: str = None,#'human'
    timelimit: float = mctf_config["max_time"], #600.,
    #logname: str = "match.log",
    q_table: QTable = None, #str = None #Not a string?! Is already a QTable!
    teamsize3: bool = False,
    ignoreseed = True,
    sim_speed = 3,
    prev_act = False
):
    # # Only enable for limited testing (do not enable for long training runs, will increase time a lot!)
    render_mode = 'human'

    #make sure seed is random #did not work because random instance was inherited from common source (train_qlearn main or imports?).
    #seed=np.random.randint(0, 100000) #TESTING IN PROGRESS: did this break randomization between parallel runs? Something did, as they are equal rn... #No, this is not the cause...
    
    # Set score function to the selected reward (match statement syntax might require python version 3.10 or newer)
    match reward_choice:
        case "caps_and_grabs":
            reward_method = caps_and_grabs
        # case "double_aggressive_rew":
        #     reward_method = double_aggressive_rew
        case "single_aggressive_rew":
            reward_method = single_aggressive_rew
        # case "defensive_rew":
        #     reward_method = defensive_rew
        case "caps_and_tags":
            reward_method = caps_and_tags
        case "single_aggressive26":
            reward_method = single_aggressive26
        case "aggressive_tags_26":
            reward_method = aggressive_tags_26
        case _:
            print("Error: Invalid reward choice. Please select a valid reward function.")
            return

    # if teamsize3: #3v3 mode is now introduced as parameter, if selected we prepare teamsize3 mode and training.
    #     config_dict["dynamics"] = ["si", "si", "si", "si", "si", "si"]
    # else:
    #     config_dict["dynamics"] = ["si", "si", "si", "si"]

    # TODO change back, Just to render a visual test match faster:
    mctf_config["sim_speedup_factor"] = sim_speed#1 #20 #for faster visual test rendering (currently set to 1, since this appears to be the setting for the mctf26 configuration)
    
    # timelimit is changed if something was specified (else mctf_config value)
    mctf_config["max_time"] = timelimit #6000.#timelimit #6000.
    mctf_config["default_init"] = False #ignoreseed# This was changed to always be random, since non-random starts (still with seed) are without merrit for our purposes.
    # The speedup factor was observed to change the rendering of the agents. Thus it does affect rendered speed (probably every frame computed is rendered?, perhaps this explains why a "10-minute game" is rendered in ~4min...).

    if teamsize3: #3v3
        #env = pyquaticus_v0.PyQuaticusEnv(team_size=3, action_space="discrete", config_dict=config_dict, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method, 'agent_4': reward_method, 'agent_5': reward_method},render_mode=render_mode)
        env = CompPyquaticusEnv(action_space="discrete", config_dict=mctf_config, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method, 'agent_4': reward_method, 'agent_5': reward_method}, render_mode=render_mode)
    else: #2v2 (not really possible in 26 environment!)
        print("Warning! Team size is not consistent with competition setting, despite launching the competition setting. This might not work properly with less than 3v3.")
        #env = pyquaticus_v0.PyQuaticusEnv(team_size=2, action_space="discrete", config_dict=config_dict, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method, 'agent_4': reward_method, 'agent_5': reward_method}, render_mode=render_mode)
        env = CompPyquaticusEnv(action_space="discrete", config_dict=mctf_config, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method}, render_mode=render_mode)
    term_g = {'agent_0':False,'agent_1':False,'agent_2':False} #this was already set for 3v3, apparently this works with 2v2? (Perhaps there is an alternative win condition to timelimit that we never hit?)
    truncated_g = {'agent_0':False,'agent_1':False,'agent_2':False}
    term = term_g
    trunc = truncated_g
    #seed = 12345 #SEED for "random" starts
    reset_opts = {'normalize_obs': False, 'normalize_state': False}
    obs, info = env.reset(options=reset_opts, seed=seed)

    # temp_captures = env.state["captures"]
    # temp_grabs = env.state["grabs"]
    # temp_tags = env.state["tags"]
    
    if teamsize3: #3v3
        # Base_combine agents
        # H_one = Heuristic_CTF_Agent('agent_3', env, mode=difficulty, continuous=False)
        # H_two = Heuristic_CTF_Agent('agent_4', env, mode=difficulty, continuous=False)            #TODO TODO continue implementing the prev_act variant
        # H_three = Heuristic_CTF_Agent('agent_5', env, mode=difficulty, continuous=False)
        H_one = BaseAttacker('agent_3', env, mode=difficulty, continuous=False)
        H_two = BaseAttacker('agent_4', env, mode=difficulty, continuous=False)
        H_three = BaseAttacker('agent_5', env, mode=difficulty, continuous=False)
        
        # print("Setting up q-learn agents")
        if q_table == None: print("Error: q-table not set up before agents are created.")
        u_table = QTable(q_table.LEARNING_RATE, q_table.DISCOUNT_FACTOR, q_table.INITIAL_Q_VALUE, prev_action=q_table.prev_action) #since only the q-table and its values are used, it is not necessary to set all parameters (like sharpturns) here. Careful though, prev_action is necessary...
        u_table.qtable = np.copy(q_table.qtable)
        # q-learn agents
        R_one = QlearnPolicy('agent_0', env, q_table, u_table)
        R_two = QlearnPolicy('agent_1', env, q_table, u_table)
        R_three = QlearnPolicy('agent_2', env, q_table, u_table)
    else: #2v2
        # Base_combine agents
        H_one = Heuristic_CTF_Agent('agent_2', env, mode=difficulty, continuous=False)
        H_two = Heuristic_CTF_Agent('agent_3', env, mode=difficulty, continuous=False)
        
        # print("Setting up q-learn agents")
        if q_table == None: print("Error: q-table not set up before agents are created.")
        u_table = QTable(q_table.LEARNING_RATE, q_table.DISCOUNT_FACTOR, q_table.INITIAL_Q_VALUE, prev_action=q_table.prev_action)
        u_table.qtable = np.copy(q_table.qtable)
        # q-learn agents
        R_one = QlearnPolicy('agent_0', env, q_table, u_table)
        R_two = QlearnPolicy('agent_1', env, q_table, u_table)

    step = 0
    rewardsteps = []   #actually, this is easier (still 2 dim but turned "90°") [[], []] #two-dimensional list to track both agents of this team
    while True:
        if teamsize3:
            # Base_combine agents
            three = H_one.compute_action(obs, info)
            four = H_two.compute_action(obs, info)
            five = H_three.compute_action(obs, info)
            
            zero = R_one.compute_action(obs, info)
            one = R_two.compute_action(obs, info)
            two = R_three.compute_action(obs, info)

            # For all three agents necessary data for q-value update is saved:
            a0_qstep = R_one.q_Table.prepareUpdate(obs, info, 'agent_0', zero)
            a1_qstep = R_two.q_Table.prepareUpdate(obs, info, 'agent_1', one)
            a2_qstep = R_three.q_Table.prepareUpdate(obs, info, 'agent_2', two)
            #(ownpos, opp1_bearing, opp2_bearing, b_flag, r_flag, action)
            
            # 3v3 step
            obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five})
            # print("heading global or not?: ",info['agent_0']['global_state'][('agent_0', "heading")])
            # print("Reward:",reward)#Reward: {'agent_0': 0.2734431070341813, 'agent_1': 0.2208806900959132, 'agent_2': -0.12809429110777018, 'agent_3': 0.0, 'agent_4': 0.0, 'agent_5': 0.0}
            # print(reward["agent_0"],reward["agent_1"],reward["agent_2"])
            
            # Update Q-Table for both agents (same table, two updates)
            # print("\nActions: zero",zero,"; one",one)
            R_one.set_q_value(a0_qstep[0], a0_qstep[1], a0_qstep[2], a0_qstep[3], a0_qstep[4], a0_qstep[5], reward['agent_0'])
            s_table[a0_qstep[0]][a0_qstep[1]][a0_qstep[2]][a0_qstep[3]][a0_qstep[4]] += 1
            R_two.set_q_value(a1_qstep[0], a1_qstep[1], a1_qstep[2], a1_qstep[3], a1_qstep[4], a1_qstep[5], reward['agent_1'])
            s_table[a1_qstep[0]][a1_qstep[1]][a1_qstep[2]][a1_qstep[3]][a1_qstep[4]] += 1
            R_three.set_q_value(a2_qstep[0], a2_qstep[1], a2_qstep[2], a2_qstep[3], a2_qstep[4], a2_qstep[5], reward['agent_2'])
            s_table[a2_qstep[0]][a2_qstep[1]][a2_qstep[2]][a2_qstep[3]][a2_qstep[4]] += 1
            rewardsteps.append({'agent_0': reward['agent_0'], 'agent_1': reward['agent_1'], 'agent_2': reward['agent_2']})
        if not teamsize3:
            # Base_combine agents
            two = H_one.compute_action(obs, info)
            three = H_two.compute_action(obs, info)
            
            zero = R_one.compute_action(obs, info)
            one = R_two.compute_action(obs, info)

            # For both agents necessary data for q-value update is saved:
            a0_qstep = R_one.q_Table.prepareUpdate(obs, info, 'agent_0', zero)
            a1_qstep = R_two.q_Table.prepareUpdate(obs, info, 'agent_1', one)
            #(ownpos, opp1_bearing, opp2_bearing, b_flag, r_flag, action)
            
            # 2v2 step
            obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three})
            #print("\n\nReward:",reward)#Reward: {'agent_0': 0.2734431070341813, 'agent_1': 0.2208806900959132, 'agent_2': -0.12809429110777018, 'agent_3': 0.0, 'agent_4': 0.0, 'agent_5': 0.0}
            
            # Update Q-Table for both agents (same table, two updates)
            #print("\nActions: zero",zero,"; one",one)
            R_one.set_q_value(a0_qstep[0], a0_qstep[1], a0_qstep[2], a0_qstep[3], a0_qstep[4], a0_qstep[5], reward['agent_0'])
            s_table[a0_qstep[0]][a0_qstep[1]][a0_qstep[2]][a0_qstep[3]][a0_qstep[4]] += 1
            R_two.set_q_value(a1_qstep[0], a1_qstep[1], a1_qstep[2], a1_qstep[3], a1_qstep[4], a1_qstep[5], reward['agent_1'])
            s_table[a1_qstep[0]][a1_qstep[1]][a1_qstep[2]][a1_qstep[3]][a1_qstep[4]] += 1
            rewardsteps.append({'agent_0': reward['agent_0'], 'agent_1': reward['agent_1']})
        # (end of teamsize if-block)

        k =  list(term.keys()) #Gameover check.

        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            # Game over
            break
    #End of while True (game loop)

    # print("\n~~~Run Concluded~~~")
    # formatted_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    # print(f" Time: {formatted_time}")
    # print("agent collisions:",env.state['agent_collisions'])
    # print("SCORE: ",env.state['captures'])
    # print("grabs: ",env.state['grabs'])
    env.close()
    return rewardsteps, env.state['captures'], env.state['grabs'], env.state['tags'], u_table 
#End of train_qlearn()