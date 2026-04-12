from datetime import datetime
#import logging
import sys
#import os
#import os.path
#import pyquaticus
import numpy as np
from numpy.typing import NDArray
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.multi_rhea_policy import MRHEA_Agent, MRHEA_Environment
from pyquaticus.base_policies.rhealg_policy2 import RHEA_Agent, RHEA_Environment
from pyquaticus.base_policies.ultra_def_policy import UltraDefender
from qtable import QlearnPolicy, QTable
from pyquaticus.utils.rewards import caps_and_grabs, defensive_rew, double_aggressive_rew, single_aggressive_rew, caps_and_tags, aggr_rew_alt, aggressive_tags_24, aggressive_oob_tags_24, caps_and_tags_oob
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
    reward_choice: str = "adjustmepls", 
    render_mode: str = None, # or 'human'
    timelimit: float = 600.,
    logname: str = "match.log",
    q_table: QTable = None, #str = None #Not a string?! Is already a QTable!
    math2: bool = False
):
    
    # Set score function to the selected reward (match statement syntax might require python version 3.10 or newer)
    match reward_choice:
        case "caps_and_grabs":
            reward_method = caps_and_grabs
        case "double_aggressive_rew":
            reward_method = double_aggressive_rew
        case "single_aggressive_rew":
            reward_method = single_aggressive_rew
        case "defensive_rew":
            reward_method = defensive_rew
        case "caps_and_tags":
            reward_method = caps_and_tags
        case "aggr_rew_alt":
            reward_method = aggr_rew_alt
        case "aggressive_oob_tags_24":
            reward_method = aggressive_oob_tags_24
        case "caps_and_tags_oob":
            reward_method = caps_and_tags_oob
        case "aggressive_tags_24":
            reward_method = aggressive_tags_24
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
    # logging.basicConfig(
    #    filename=logname,
    #    filemode="w",   #"w" to overwrite, "a" to append. Does it overwrite within the loop? if so, a.
    #    level=logging.INFO,
    #    format="%(asctime)s %(message)s",
    #    force=True
    # )

    env = pyquaticus_v0.PyQuaticusEnv(team_size=2, action_space="discrete", config_dict=config_dict, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method},
    render_mode=render_mode) #'human')  #None)#'human')
    term_g = {'agent_0':False,'agent_1':False,'agent_2':False}
    truncated_g = {'agent_0':False,'agent_1':False,'agent_2':False}
    term = term_g
    trunc = truncated_g
    #seed = 12345 #SEED for "random" starts
    reset_opts = {'normalize_obs': False, 'normalize_state': False}
    obs, info = env.reset(options=reset_opts, seed=seed)

    # temp_captures = env.state["captures"]
    # temp_grabs = env.state["grabs"]
    # temp_tags = env.state["tags"]
    

    # Base_combine agents
    H_one = Heuristic_CTF_Agent('agent_2', env, mode=difficulty, continuous=False)#TODO try if False works (seems more fair)
    H_two = Heuristic_CTF_Agent('agent_3', env, mode=difficulty, continuous=False)
    
    # print("Setting up q-learn agents")
    if q_table == None: print("Error: q-table not set up before agents are created.")
    if math2:
        u_table = None
    else:
        u_table = QTable(q_table.LEARNING_RATE, q_table.DISCOUNT_FACTOR, q_table.INITIAL_Q_VALUE)
        u_table.qtable = np.copy(q_table.qtable)
    R_one = QlearnPolicy('agent_0', env, q_table)
    R_two = QlearnPolicy('agent_1', env, q_table)

    step = 0
    rewardsteps = []   #actually, this is easier (still 2 dim but turned "90°") [[], []] #two-dimensional list to track both agents of this team #TODO check if rewardcurve is meaningless now
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
        
        # Update Q-Table for both agents (same table, two updates) NOTE the qtable is only updated after the game (batch). But necessary information is saved here
        a0_step2 = R_one.q_Table.prepareUpdate(obs, 'agent_0', 4) #action 4 is not being used, but 4 so there is no "error" printed.
        a1_step2 = R_two.q_Table.prepareUpdate(obs, 'agent_1', 4)
        #print("\nActions: zero",zero,"; one",one)

        R_one.set_q_value(a0_step2[0], a0_step2[1], a0_step2[2], a0_step2[3], a0_step2[4], a0_qstep[5], reward['agent_0']) #only action is used from previous frame/timestep
        s_table[a0_qstep[0]][a0_qstep[1]][a0_qstep[2]][a0_qstep[3]][a0_qstep[4]] += 1
        R_two.set_q_value(a1_step2[0], a1_step2[1], a1_step2[2], a1_step2[3], a1_step2[4], a1_qstep[5], reward['agent_1']) #only action is used from previous frame/timestep
        s_table[a1_qstep[0]][a1_qstep[1]][a1_qstep[2]][a1_qstep[3]][a1_qstep[4]] += 1
        
        # Keep track of reward
        rewardsteps.append({'agent_0': reward['agent_0'], 'agent_1': reward['agent_1']})
        # print("REWARD:", {'agent_0': reward['agent_0'], 'agent_1': reward['agent_1']})
        # -Logging utility- (disabled for training, too much memory)
        # Writes the gamestate info into pyquaticus/match.log #this seems like it is doubled? 
        #logging.info("obs: %s", obs) 
        #logging.info("reward: %s", reward)
        #logging.info("info: %s", info)

        k =  list(term.keys()) #Gameover check.

        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            # Game over
            #scores.append(env.state['captures']) #scores for both teams at the end of each episode, for all episodes TODO not for all episodes, perhaps a avg for a number of episodes because memory
            #grabslist.append(env.state['grabs']) #grabs too #is done outside of train now
            break
    # These are some statistics we are exporting: Actually, these lines of code seem not necessary
    # for i in range(len(env.state["captures"])):
    #     temp_captures[i] += env.state["captures"][i]
    # for i in range(len(env.state["grabs"])):
    #     temp_grabs[i] += env.state["grabs"][i]
    # for i in range(len(env.state["tags"])):
    #     temp_tags[i] += env.state["tags"][i]

    # print("\n~~~Run Concluded~~~")
    # formatted_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    # print(f" Time: {formatted_time}")
    # print("agent collisions:",env.state['agent_collisions'])
    # print("SCORE: ",env.state['captures'])
    # print("grabs: ",env.state['grabs'])
    env.close()
    # t1 = 0.
    # t2 = 0.
    # for e in rewardsteps:
    #     t1 += e['agent_0']
    #     t2 += e['agent_1']
    # print(f"t1: {t1}   t2: {t2}")
    return rewardsteps, env.state['captures'], env.state['grabs'], env.state['tags'], u_table 
#End of train_qlearn()

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
#End of visualize_reward_curve()

if __name__ == "__main__":
    # scores = np.load("qtrainlog/lrate0.1_discount0.9_initialq0.0_caps_and_tags_scores.npy", allow_pickle=True)
    # print(scores)
    # print(f"Scores shape: {scores.shape}")
    # print(f"Scores ndim: {scores.ndim}")
    # sys.exit(0)
    
    # Prepared experiments are made easier to launch (editor performance is affected once some of these are launched, and they are made to be processed simultaneously)
    if len(sys.argv) > 1:
        #if argument 1 set rewardchoice, etc to x
        if sys.argv[1] == "1": #Set to batch 5
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew_bicheck1"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "2":#Set to batch 5
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew_bicheck2"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "3":#Set to batch 5
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew_bicheck3"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "4":#Set to batch 5
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew_bicheck4"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "5":#set to batch 5
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew_bicheck5"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "6":
            rewardchoice = "caps_and_tags"
            filename_suffix = "lrate0.2_discount0.9_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.2, 0.9, 10.0
        elif sys.argv[1] == "7":
            rewardchoice = "caps_and_tags"
            filename_suffix = "lrate0.2_discount0.95_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.2, 0.95, 10.0
        elif sys.argv[1] == "8":
            rewardchoice = "caps_and_tags"
            filename_suffix = "lrate0.2_discount0.85_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.2, 0.85, 10.0
        elif sys.argv[1] == "9":
            rewardchoice = "caps_and_tags"
            filename_suffix = "lrate0.15_discount0.9_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.15, 0.9, 10.0
        elif sys.argv[1] == "10":
            rewardchoice = "single_aggressive_rew" #with pre-training
            filename_suffix = "pretrained_lrate0.1_discount0.9_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "11":
            rewardchoice = "caps_and_tags"
            filename_suffix = "pretrained_lrate0.1_discount0.9_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
        elif sys.argv[1] == "test":
            """
            Input must be: >python qlearn_test.py test lrate0.1_discount0.9_initialq0.0_ single_aggressive_rew
            (where filename suffix was lrate0.1_discount0.9_initialq0.0_single_aggressive_rew )
            Execute in correct folder so file (with qtable) can be found. 
            """
            rewardchoice = sys.argv[3]
            filename_suffix = sys.argv[2] + rewardchoice
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 0.0
            print("Manually testing qtable policy with rendering enabled.")
            qt = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, ("qtrainlog/" + filename_suffix + "_q_table.npy"))
            st = np.zeros((4, 4, 4, 2, 2), dtype=np.int32)
            train_qlearn(st, seed=0, difficulty="easy", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)
            sys.exit(0)
        else:
            print("!Wrong rewardchoice argument!")
            rewardchoice = "caps_and_tags"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
            print("Manually testing qtable policy with rendering enabled.")
            qt = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, ("qtrainlog/" + filename_suffix + "_q_table.npy"))
            st = np.zeros((4, 4, 4, 2, 2), dtype=np.int32)
            train_qlearn(st, seed=0, difficulty="easy", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)
            sys.exit(0)
    else:
        print("No rewardchoice given")
        rewardchoice = "single_aggressive_rew"
        filename_suffix = "batch 2 hard/vshard_lrate0.1_discount0.95_initialq10.0_single_aggressive_rew"
        LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.95, 10.0
        print("Manually testing qtable policy with rendering enabled.")
        qt = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, ("qtrainlog/" + filename_suffix + "_q_table.npy"))
        st = np.zeros((4, 4, 4, 2, 2), dtype=np.int32)
        train_qlearn(st, seed=0, difficulty="hard", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)
        sys.exit(0)
    #rewardchoice = "single_aggressive_rew"
    #rewardchoice = "double_aggressive_rew"
    #rewardchoice = "caps_and_grabs"
    #rewardchoice = "caps_and_tags"

    "--------------------------------------------"
    # Creating filenames for the saved q-table and reward curve, with some of the training parameters included in the name for better tracking
    #filename_suffix = f"{rewardchoice}_neutral" 
    #filename_suffix = ""
    "--------------------------------------------"
    filename_suffix = "qtrainlog/batch 5/"+filename_suffix 
    
    # Create qtrainlog directory if it doesn't exist
    #os.makedirs("qtrainlog", exist_ok=True) #should exist, except if started from wrong folder...

    # Plot reward curve from file 
    #visualize_reward_curve("reward_curve_aggr_easy_130i_neutral.npy")
    
    # Print Q-Table from file
    #qtablo = QTable("q_table_aggr_hard_overnight_neutral_rew.npy")
    #print(qtablo.qtable)

    # Run training loop for multiple iterations (one setting, repeated with the same qtable)
    # print("Setting up Q-Table")
    qtableee = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE)
    s_table = np.zeros((4, 4, 4, 2, 2), dtype=np.uint32) #statecount-table
    # same dimensionality as qtable, but no action-options (because we just want to know about the state... for now)
    # statecount-table (to measure how many times a state was updated)
    rewardcurve = [] #is created by the 
    scorelist = []
    grabslist = []
    tagslist = []
    index = 0 
    for i in range(500): #set batch 5
    #while datetime.now().hour < 11 or datetime.now().hour > 20: #train until 1 am, then save the q-table and reward curve (TODO visualize the reward cuve later)
        # print("Beginning training run at time ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        seeed = np.random.randint(0, 100000) #random seed while training, set of seeds when testing (TODO)
        #logstructure = []
        if index < 500 or True: #pretraininng with easy opponents, for more exploration on opponent base  [pretraining"easy" disabled for now, all training against easy(now hard)]
            rewardsteps, capture_entry, grab_entry, tag_entry = train_qlearn(s_table, seed=seeed, difficulty="hard", reward_choice=rewardchoice, render_mode=None, timelimit=600., q_table=qtableee)
            # tags, rewardlist, captures, grabs are all for [0] and [1] (the two teams)
            # After each episode update the values of q-table. For this purpose updates are calculated during the episode into the u-table. Now it gets switched with q-table:
            #qtableee.qtable = u_table.qtable
            qtableee.avgQUpdate()
        else:
            rewardsteps, capture_entry, grab_entry, tag_entry = train_qlearn(s_table, seed=seeed, difficulty="hard", reward_choice=rewardchoice, render_mode=None, timelimit=600., q_table=qtableee)
            #qtableee.qtable = u_table.qtable
            qtableee.avgQUpdate()

        # Some of the data we are tracking needs to be added to another list structure:
        #rewardcurve.append(rewardsteps)#wrong, need to sum it up exactly before that
        rewardsum = [0., 0.]
        #print("\n\n -##########- \grab_entry: ",grab_entry)
        for r in rewardsteps:
            # agent0 and agent1 are the ones where reward is interesting for us.
            rewardsum[0] += r['agent_0']
            rewardsum[1] += r['agent_1']
        rewardcurve.append(rewardsum) 
        # rewardcurve thus contains agents 0 & 1 as index [0] & [1].
        scorelist.append(capture_entry) #zB [ 0 21]
        grabslist.append(grab_entry) #zB [ 0 23]
        tagslist.append(tag_entry) #zB [ 2 16]

        # Print all important data (especially the q-table!) regularly to file:
        if (index % 50) == 49: 
            # print(f"(Pre-storing q-table to file \"{filename_suffix}_q_table.npy\" at index {index}.)")
            qtableee.toFile(f"{filename_suffix}_q_table.npy")
            # print(f"(Pre-storing rewardcurve to file \"{filename_suffix}_reward_curve.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_reward_curve.npy", rewardcurve)
            #print("Storing logstructure to file", "logstructure.npy")
            #np.save(f"{filename_suffix}_logstructure.npy", logstructure) 
            # print(f"(Pre-storing scores to file \"{filename_suffix}_scores.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_scores.npy", scorelist) 
            # print(f"(Pre-storing grabslist to file \"{filename_suffix}_grabslist.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_grabslist.npy", grabslist)
            # print(f"(Pre-storing tagslist to file \"{filename_suffix}_tagslist.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_tagslist.npy", tagslist) 
            # print(f"(Pre-storing statecount-table to file \"{filename_suffix}_statecount.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_statecount.npy", s_table) 
            #print("Statecount table: ",s_table)
        # Print qtable regularly as checkpoint to additional file (but not too oft because memory leak)
        if (index % 500) == 499: 
            # print(f"(In-between-storing q-table to file \"{filename_suffix}_q_table_i{index}.npy\".)")
            qtableee.toFile(f"{filename_suffix}_q_table_i{index}.npy")

        #np.save(f"{filename_suffix}_logstructure{index}.npy", logstructure) 
        # discard logstructure now, so memory does not leak
        #logstructure = []
        index += 1
        # print(f"Completed training run {index}")

    # Epilog (saving q-table and reward curve to file)
    # print(f"Storing q-table to file \"{filename_suffix}_q_table.npy\".")
    qtableee.toFile(f"{filename_suffix}_q_table.npy") #Hmm. Do we need a better naming system, some way to keep track of trained  policies (maybe even in thesis? certainly in slides...), better way to automatically name things, actual pipeline in general
    #testqtable = QTable("q_table.npy")
    # print(f"Storing rewardcurve to file \"{filename_suffix}_reward_curve.npy\".")
    np.save(f"{filename_suffix}_reward_curve.npy", rewardcurve)
    #print("Storing logstructure to file", "logstructure.npy")
    #np.save(f"{filename_suffix}_logstructure.npy", logstructure) 
    # print(f"Storing scorelist to file \"{filename_suffix}_scores.npy\".")
    np.save(f"{filename_suffix}_scores.npy", scorelist)
    # print(f"Storing grabslist to file \"{filename_suffix}_grabslist.npy\".")
    np.save(f"{filename_suffix}_grabslist.npy", grabslist)
    # print(f"Storing tagslist to file \"{filename_suffix}_tagslist.npy\".")
    np.save(f"{filename_suffix}_tagslist.npy", tagslist)