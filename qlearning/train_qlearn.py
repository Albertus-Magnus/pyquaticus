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
from pyquaticus.utils.rewards import caps_and_grabs, defensive_rew, double_aggressive_rew, single_aggressive_rew, caps_and_tags
from qlearn_test import train_qlearn, visualize_reward_curve
#from multiprocessing import Pool, Value, Lock #i don't need any parallel processing (is qlearn even compatible?), i just need to run 10 scripts in different terminals...

"""
This was written to train multiples of a given 
parameter setting (or multiple settings) in parallel 
and for each create a test of 2x qlearn policy agents 
versus 2x base_combined agents during the Masters 
thesis (March 2026).
"""

# this class is just there to select (and store for some time) the right parameters and generate filenames (to log the data reliably)
class ParameterSet:
    #def __init__(self, rewardchoice: str, lrate: float, discount: float, initialq: float, pretrain: bool, name: str, fodler: str, index: int): #test first without type
    def __init__(self, rewardchoice, lrate, discount, initialq, pretrain, name, folder, index):
        self.rewardchoice = rewardchoice #zB "single_aggressive_rew"
        self.foldername = folder # "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew_bicheck1"
        self.LEARNING_RATE, self.DISCOUNT_FACTOR, self.INITIAL_Q_VALUE = lrate, discount, initialq # zB 0.1, 0.9, 10.0
        self.pretrain = pretrain
        self.name = name
        self.index = index #Do we need to store the index here?

    # Create a string for file storage that contains all important info about parameters (as well as an index if parameters are used more than once).
    def create_name(self):
        if self.pretrain:
            pre = "pretrained"
        else:
            pre = "no_pre"
        # the name of all files (qtable, stats, s-table,...):   ("_qtable" etc are appended)
        n = self.name + "_" + str(self.rewardchoice) + "_lrate"+ self.LEARNING_RATE + "_discount"+self.DISCOUNT_FACTOR + "_initq" + self.INITIAL_Q_VALUE+  "_" + pre + "_nr" + self.index
        return n

if __name__ == "__main__":
    # scores = np.load("qtrainlog/lrate0.1_discount0.9_initialq0.0_caps_and_tags_scores.npy", allow_pickle=True)
    # print(scores)
    # print(f"Scores shape: {scores.shape}")
    # print(f"Scores ndim: {scores.ndim}")
    # sys.exit(0)

    if len(sys.argv) > 1:
        # Selecting preset of training parameters ("train" to make sure the files are not overwritten by mistake)
        parametersets = []
        if sys.argv[1] == "train":
            # Do large batch training here.
            #########################################
            for i in range(20):
                parametersets.append(ParameterSet(0.1, 0.9, 10.0, False, "avgtest1", "qtrainlog/batch 6 avg/"))
            for i in range(20):
                parametersets.append()
            #########################################
        else:
            # Do visual test match here.
            setup = ParameterSet(0.1, 0.9, 10.0, False, "example", "qtrainlog/example_folder/") #lrate: Any, discount: Any, initialq: Any, pretrain: Any, name: Any, folder)
            # (do i need to change this so it loads a file?) prolly, 'cause it is for visual test match of trained policy
            # run these settings with 'human' rendering TODO
            print("No rewardchoice given")
            rewardchoice = "single_aggressive_rew"
            filename_suffix = "batch 2 hard/vshard_lrate0.1_discount0.95_initialq10.0_single_aggressive_rew"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.95, 10.0
            print("Manually testing qtable policy with rendering enabled.")
            qt = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, ("qtrainlog/" + filename_suffix + "_q_table.npy"))
            st = np.zeros((4, 4, 4, 2, 2), dtype=np.int8)
            train_qlearn(st, seed=0, difficulty="hard", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)
            sys.exit(0)


    
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
            st = np.zeros((4, 4, 4, 2, 2), dtype=np.int8)
            train_qlearn(st, seed=0, difficulty="easy", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)
            sys.exit(0)
        else:
            print("!Wrong rewardchoice argument!")
            rewardchoice = "caps_and_tags"
            filename_suffix = "lrate0.1_discount0.9_initialq10.0_caps_and_tags"
            LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0
            print("Manually testing qtable policy with rendering enabled.")
            qt = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, ("qtrainlog/" + filename_suffix + "_q_table.npy"))
            st = np.zeros((4, 4, 4, 2, 2), dtype=np.int8)
            train_qlearn(st, seed=0, difficulty="easy", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)
            sys.exit(0)
    else:
        print("No rewardchoice given")
        rewardchoice = "single_aggressive_rew"
        filename_suffix = "batch 2 hard/vshard_lrate0.1_discount0.95_initialq10.0_single_aggressive_rew"
        LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.95, 10.0
        print("Manually testing qtable policy with rendering enabled.")
        qt = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, ("qtrainlog/" + filename_suffix + "_q_table.npy"))
        st = np.zeros((4, 4, 4, 2, 2), dtype=np.int8)
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
    # If using existing q-table, load from file
    tablefromfile = False
    if tablefromfile:
        print("Loading Q-Table from file")
        qtableee = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, filename_suffix)#TODO filename_suffix correct?
    else:
        print("Setting up Q-Table")
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
    #while datetime.now().hour < 11 or datetime.now().hour > 20: #train until 1 am, then save the q-table and reward curve
        print("Beginning training run at time ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        seeed = np.random.randint(0, 100000) #random seed while training, set of seeds when testing (TODO)
        #logstructure = []
        if index < 500 or True: #pretraininng with easy opponents, for more exploration on opponent base  [pretraining"easy" disabled for now, all training against easy(now hard)]
            rewardsteps, capture_entry, grab_entry, tag_entry, u_table = train_qlearn(s_table, seed=seeed, difficulty="hard", reward_choice=rewardchoice, render_mode=None, timelimit=600., q_table=qtableee)
            # tags, rewardlist, captures, grabs are all for [0] and [1] (the two teams)
            # After each episode update the values of q-table. For this purpose updates are calculated during the episode into the u-table. Now it gets switched with q-table:
            qtableee.qtable = u_table.qtable
        else:
            rewardsteps, capture_entry, grab_entry, tag_entry, u_table = train_qlearn(s_table, seed=seeed, difficulty="hard", reward_choice=rewardchoice, render_mode=None, timelimit=600., q_table=qtableee)
            qtableee.qtable = u_table.qtable

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
        if (index % 50) == 0: 
            print(f"(Pre-storing q-table to file \"{filename_suffix}_q_table.npy\" at index {index}.)")
            qtableee.toFile(f"{filename_suffix}_q_table.npy")
            print(f"(Pre-storing rewardcurve to file \"{filename_suffix}_reward_curve.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_reward_curve.npy", rewardcurve)
            #print("Storing logstructure to file", "logstructure.npy")
            #np.save(f"{filename_suffix}_logstructure.npy", logstructure) 
            print(f"(Pre-storing scores to file \"{filename_suffix}_scores.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_scores.npy", scorelist) 
            print(f"(Pre-storing grabslist to file \"{filename_suffix}_grabslist.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_grabslist.npy", grabslist)
            print(f"(Pre-storing tagslist to file \"{filename_suffix}_tagslist.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_tagslist.npy", tagslist) 
            print(f"(Pre-storing statecount-table to file \"{filename_suffix}_statecount.npy\" at index {index}.)")
            np.save(f"{filename_suffix}_statecount.npy", s_table) 
            #print("Statecount table: ",s_table)
        # Print qtable regularly as checkpoint to additional file (but not too oft because memory leak)
        if (index % 500) == 0: 
            print(f"(In-between-storing q-table to file \"{filename_suffix}_q_table_i{index}.npy\".)")
            qtableee.toFile(f"{filename_suffix}_q_table_i{index}.npy")

        #np.save(f"{filename_suffix}_logstructure{index}.npy", logstructure) 
        # discard logstructure now, so memory does not leak
        #logstructure = []
        index += 1
        print(f"Completed training run {index}")

    # Epilog (saving q-table and reward curve to file)
    print(f"Storing q-table to file \"{filename_suffix}_q_table.npy\".")
    qtableee.toFile(f"{filename_suffix}_q_table.npy") #Hmm. Do we need a better naming system, some way to keep track of trained  policies (maybe even in thesis? certainly in slides...), better way to automatically name things, actual pipeline in general
    #testqtable = QTable("q_table.npy")
    print(f"Storing rewardcurve to file \"{filename_suffix}_reward_curve.npy\".")
    np.save(f"{filename_suffix}_reward_curve.npy", rewardcurve)
    #print("Storing logstructure to file", "logstructure.npy")
    #np.save(f"{filename_suffix}_logstructure.npy", logstructure) 
    print(f"Storing scorelist to file \"{filename_suffix}_scores.npy\".")
    np.save(f"{filename_suffix}_scores.npy", scorelist)
    print(f"Storing grabslist to file \"{filename_suffix}_grabslist.npy\".")
    np.save(f"{filename_suffix}_grabslist.npy", grabslist)
    print(f"Storing tagslist to file \"{filename_suffix}_tagslist.npy\".")
    np.save(f"{filename_suffix}_tagslist.npy", tagslist)