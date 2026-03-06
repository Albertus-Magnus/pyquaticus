from datetime import datetime
import logging
import sys
import os
import os.path
import pyquaticus
import numpy as np
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.multi_rhea_policy import MRHEA_Agent, MRHEA_Environment
from pyquaticus.base_policies.rhealg_policy2 import RHEA_Agent, RHEA_Environment
from pyquaticus.base_policies.ultra_def_policy import UltraDefender
from qtable import QlearnPolicy, QTable
from pyquaticus.utils.rewards import caps_and_grabs, aggressive_rew, defensive_rew, double_aggressive_rew, single_aggressive_rew, caps_and_tags
#from multiprocessing import Pool, Value, Lock #i don't need any parallel processing (is qlearn even compatible?), i just need to run 10 scripts in different terminals...

"""
This was copied from magnus_test.py and modified to 
create a test of 2x qlearn policy agents versus 
2x base_combined agents during the Masters thesis (Feb 2026).
"""

# To run the training this is called as a function, with MODE and reward function as parameters.

def train_qlearn(
    rewardcurve,
    scores,
    grabslist,
    seed: int = 12345,
    # seed for "random" starts
    difficulty: str = "hard",
    # difficulty is the MODE of the example agents, can be "hard", "medium" or "easy"
    reward_choice: str = "adjustmepls", #maybe could be string, but cleaner so?
    render_mode: str = None,#'human'
    timelimit: float = 600.,
    logname: str = "match.log",
    q_table: str = None
):
    
    # Set score function to the selected reward (match statement syntax might require python version 3.10 or newer)
    match reward_choice:
        case "caps_and_grabs":
            reward_method = caps_and_grabs
        case "double_aggressive_rew":
            reward_method = double_aggressive_rew
        case "single_aggressive_rew":
            reward_method = single_aggressive_rew
        case "aggressive_rew":
            reward_method = aggressive_rew
        case "defensive_rew":
            reward_method = defensive_rew
        case "caps_and_tags":
            reward_method = caps_and_tags
        case _:
            print("Error: Invalid reward choice. Please select a valid reward function.")
            return

    config_dict = {}
    config_dict["max_time"] = timelimit#600.0
    config_dict["max_score"] = 100
    config_dict["render_agent_ids"] = True
    config_dict["dynamics"] = ["si", "si", "si", "si"
                               ]
    config_dict["sim_speedup_factor"] = 3
    config_dict["default_init"] = False #random starting positions (uses seed)

    #-Logging utility-
    logging.basicConfig(
        filename=logname,
        filemode="w",   #"w" to overwrite, "a" to append. Does it overwrite within the loop? if so, a.
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True
    )

    env = pyquaticus_v0.PyQuaticusEnv(team_size=2, action_space="discrete", config_dict=config_dict, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method},
    render_mode=render_mode) #'human')  #None)#'human')
    term_g = {'agent_0':False,'agent_1':False,'agent_2':False}
    truncated_g = {'agent_0':False,'agent_1':False,'agent_2':False}
    term = term_g
    trunc = truncated_g
    #seed = 12345 #SEED for "random" starts
    reset_opts = {'normalize_obs': False, 'normalize_state': False}
    obs, info = env.reset(options=reset_opts, seed=seed)

    temp_captures = env.state["captures"]
    temp_grabs = env.state["grabs"]
    temp_tags = env.state["tags"]
    

    # Base_combine agents
    H_one = Heuristic_CTF_Agent('agent_2', env, mode=difficulty, continuous=False)#TODO try if False works (seems more fair)
    H_two = Heuristic_CTF_Agent('agent_3', env, mode=difficulty, continuous=False)
    
    print("Setting up q-learn agents")
    if q_table == None: print("Error: q-table not set up before agents are created.")
    R_one = QlearnPolicy('agent_0', env, q_table)
    R_two = QlearnPolicy('agent_1', env, q_table)

    step = 0
    #rewardcurve = []
    while True:
        # Base_combine agents
        two = H_one.compute_action(obs, info)
        three = H_two.compute_action(obs, info)
        
        zero = R_one.compute_action(obs, info)
        one = R_two.compute_action(obs, info)

        # For both agents necessary data for q-value update is saved:
        a0_qstep = R_one.q_Table.prepareUpdate(obs, 'agent_0', zero)
        a1_qstep = R_two.q_Table.prepareUpdate(obs, 'agent_1', one)
        #(ownpos, opp1_bearing, opp2_bearing, b_flag, r_flag, action)
        
        # 2v2 step
        obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three})
        #print("\n\nReward:",reward)#Reward: {'agent_0': 0.2734431070341813, 'agent_1': 0.2208806900959132, 'agent_2': -0.12809429110777018, 'agent_3': 0.0, 'agent_4': 0.0, 'agent_5': 0.0}

        #}) #TODO save tag-counts
        
        # Update Q-Table for both agents (same table, two updates)
        #print("\nActions: zero",zero,"; one",one)
        R_one.q_Table.set_q_value(a0_qstep[0], a0_qstep[1], a0_qstep[2], a0_qstep[3], a0_qstep[4], a0_qstep[5], reward['agent_0'])
        R_two.q_Table.set_q_value(a1_qstep[0], a1_qstep[1], a1_qstep[2], a1_qstep[3], a1_qstep[4], a1_qstep[5], reward['agent_1'])
        # Keep track of reward (TODO need to get an underlying curve and visualize it for full training)
        rewardcurve.append(reward)
        # -Logging utility- (disabled for training, too much memory)
        # Writes the gamestate info into pyquaticus/match.log #this seems like it is doubled? 
        #logging.info("obs: %s", obs) 
        #logging.info("reward: %s", reward)
        #logging.info("info: %s", info)

        k =  list(term.keys()) #what is k? Likely a gameover check.

        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            scores.append(env.state['captures']) #scores for both teams at the end of each episode, for all episodes TODO not for all episodes, perhaps a avg for a number of episodes because memory
            grabslist.append(env.state['grabs']) #grabs too
            break
    # These are some statistics we are exporting:
    for i in range(len(env.state["captures"])):
        temp_captures[i] += env.state["captures"][i]
    for i in range(len(env.state["grabs"])):
        temp_grabs[i] += env.state["grabs"][i]
    for i in range(len(env.state["tags"])):
        temp_tags[i] += env.state["tags"][i]

    for i in range(len(env.state["captures"])):
        temp_captures[i] += env.state["captures"][i]
    for i in range(len(env.state["grabs"])):
        temp_grabs[i] += env.state["grabs"][i]
    for i in range(len(env.state["tags"])):
        temp_tags[i] += env.state["tags"][i]

    print("\n~~~Run Concluded~~~")
    formatted_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    print(f" Time: {formatted_time}")
    print("agent collisions:",env.state['agent_collisions'])
    print("SCORE: ",env.state['captures'])
    print("grabs: ",env.state['grabs'])
    env.close()
    return rewardcurve

def visualize_reward_curve(reward_curve_file):
    import matplotlib.pyplot as plt
    reward_curve = np.load(reward_curve_file, allow_pickle=True)
    agent_0_rewards = [step['agent_0'] for step in reward_curve]
    agent_1_rewards = [step['agent_1'] for step in reward_curve]
    plt.figure(figsize=(12, 6))
    plt.plot(agent_0_rewards, label='Agent 0 Rewards')
    plt.plot(agent_1_rewards, label='Agent 1 Rewards')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #if argument 1 set rewardchoice to x
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "2":
            rewardchoice = "double_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_double_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "3":
            rewardchoice = "caps_and_grabs"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_caps_and_grabs"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "4":
            rewardchoice = "caps_and_tags"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "5":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.8_discount0.9_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.8, 0.9, 10.0
        elif sys.argv[1] == "6":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.95_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.95, 10.0
        elif sys.argv[1] == "7":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.85_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.85, 10.0
        elif sys.argv[1] == "8":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.5_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.5, 10.0
        elif sys.argv[1] == "9":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.2_discount0.95_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.2, 0.95, 10.0
        elif sys.argv[1] == "10":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq100.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 100.0
        elif sys.argv[1] == "11":
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq0.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 0.0
        else:
            print("!Wrong rewardchoice argument!")
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "example_suffix"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
    else:
        print("No rewardchoice given")
        rewardchoice = "single_aggressive_rew"
        filename_suffix = "example_suffix"
        LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
    #rewardchoice = "single_aggressive_rew"
    #rewardchoice = "double_aggressive_rew"
    #rewardchoice = "caps_and_grabs"
    #rewardchoice = "caps_and_tags"

    "--------------------------------------------"
    # Creating filenames for the saved q-table and reward curve, with some of the training parameters included in the name for better tracking
    #filename_suffix = f"{rewardchoice}_neutral" #TODO better naming system, some way to keep track of trained  policies (maybe even in thesis? certainly in slides...), better way to automatically name things, actual pipeline in general
    #filename_suffix = ""
    "--------------------------------------------"
    filename_suffix = "qtrainlog/"+filename_suffix
    
    # Create qtrainlog directory if it doesn't exist
    #os.makedirs("qtrainlog", exist_ok=True) #should exist, except if started from wrong folder...

    # Plot reward curve from file 
    #visualize_reward_curve("reward_curve_aggr_easy_130i_neutral.npy")
    
    # Print Q-Table from file
    #qtablo = QTable("q_table_aggr_hard_overnight_neutral_rew.npy")
    #print(qtablo.qtable)

    # Run training loop for multiple iterations (one setting, repeated with the same qtable)
    print("Setting up Q-Table")
    qtableee = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE)
    rewardcurve = []
    scores = []
    grabslist = []
    index = 0       #right now set for 6h training
    #for i in range(130):
    while datetime.now().hour < 11 or datetime.now().hour > 20: #train until 1 am, then save the q-table and reward curve (TODO visualize the reward cuve later)
        print("Beginning training run at time ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        seeed = np.random.randint(0, 100000) #random seed while training, set of seeds when testing (TODO)
        #logstructure = []
        if index < 500 or True: #pretraininng with easy opponents, for more exploration on opponent base  [pretraining disabled for now, all training against easy]
            train_qlearn(rewardcurve, scores, grabslist, seed=seeed, difficulty="easy", reward_choice=rewardchoice, render_mode=None, timelimit=600., q_table=qtableee)
        else:
            train_qlearn(rewardcurve, scores, grabslist, seed=seeed, difficulty="hard", reward_choice=rewardchoice, render_mode=None, timelimit=600., q_table=qtableee)
        #np.save(f"{filename_suffix}_logstructure{index}.npy", logstructure) 
        # discard logstructure now, so memory does not leak
        #logstructure = []
        index += 1
        print(f"Completed training run {index}")

    # Epilog (saving q-table and reward curve to file)
    print("Storing q-table to file", "q_table.npy")
    qtableee.toFile(f"{filename_suffix}_q_table.npy") #TODO better naming system, some way to keep track of trained  policies (maybe even in thesis? certainly in slides...), better way to automatically name things, actual pipeline in general
    #testqtable = QTable("q_table.npy")
    print("Storing reward curve to file", "reward_curve.npy")
    np.save(f"{filename_suffix}_reward_curve.npy", rewardcurve)
    #print("Storing logstructure to file", "logstructure.npy")
    #np.save(f"{filename_suffix}_logstructure.npy", logstructure) 
    print("Storing scores to file", "scores.npy")
    np.save(f"{filename_suffix}_scores.npy", scores)
    print("Storing grabslist to file", "grabslist.npy")
    np.save(f"{filename_suffix}_grabslist.npy", grabslist)
    if False:
        # code to run a test with rendering 
        qtablo = QTable(f"{filename_suffix}_q_table.npy")
        rewardcurve = []
        scores = []
        grabslist = []
        train_qlearn(rewardcurve, scores, grabslist, seed=12345, difficulty="easy", reward_choice=2, render_mode='human', timelimit=600., q_table=qtablo)