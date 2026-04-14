from datetime import datetime
#import logging
from functools import partial
import os
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
from pyquaticus.utils.rewards import caps_and_grabs, defensive_rew, double_aggressive_rew, single_aggressive_rew, caps_and_tags, aggressive_tags
from qlearn_test import train_qlearn, visualize_reward_curve
from multiprocessing import Pool, Lock, Value #i don't need any parallel processing (is qlearn even compatible?), i just need to run 10 scripts in different terminals...

lock = Lock()
counter = Value('i', 0)
# number_jobs = -1
# counter = -1

"""
(used code from qlearn_test.py, also imported some)
This was written to train multiples of a given 
parameter setting (or multiple settings) in parallel 
and for each create a test of 2x qlearn policy agents 
versus 2x base_combined agents during the Masters 
thesis (March 2026).
"""

# this class is just there to select (and store for some time) the right parameters and generate filenames (to log the data reliably)
class ParameterSet:
    #def __init__(self, rewardchoice: str, lrate: float, discount: float, initialq: float, pretrain: bool, name: str, fodler: str, index: int): #test first without type
    def __init__(self, rewardchoice, dif, lrate, discount, initialq, pretrain, name, folder, index, boolchange=True, nrs=20, ep=1000):
        self.rewardchoice = rewardchoice #zB "single_aggressive_rew"
        self.dif = dif
        self.foldername = folder # "lrate0.1_discount0.9_initialq10.0_single_aggressive_rew_bicheck1"
        self.LEARNING_RATE, self.DISCOUNT_FACTOR, self.INITIAL_Q_VALUE = lrate, discount, initialq # zB 0.1, 0.9, 10.0
        self.pretrain = pretrain
        self.name = name
        self.index = index #Do we need to store the index here? (it is so if training different parametersets it will be per set and not an overarching index)
        self.boolchange = boolchange
        self.nrs = nrs
        self.ep = ep

    # Create a string for file storage that contains all important info about parameters (as well as an index if parameters are used more than once).
    def create_name(self):
        if self.pretrain:
            pre = "500-pretrained" 
        else:
            pre = "no_pre"
        # the name of all files (qtable, stats, s-table,...):   ("_qtable" etc are appended)
        #n = self.name + "_" + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_" + "1000ep_" + pre + "_nr" + str(self.index) #old name scheme, before nrs and ep were parameterized
        n = self.name + "_" + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_"+str(self.nrs)+"nrs_" + str(self.ep) + "ep_" + pre + "_nr" + str(self.index)
        #number of episodes (and 500-pretrained?) has to be hand-adjusted, perhaps change that...
        return n
    
    # Create a string for file storage that contains all important info about parameters (as well as an index if parameters are used more than once).
    def create_name_without_index(self):
        if self.pretrain:
            pre = "500-pretrained" #NOTE this 500- is missing from the first set of runs (avgtest1?)
        else:
            pre = "no_pre"
        # the name of all files (qtable, stats, s-table,...):   ("_qtable" etc are appended)
        #n = self.name + "_" + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_" + "1000ep_" + pre #old name scheme, before nrs and ep (batch 10c and before)
        n = self.name + "_" + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_" +str(self.nrs)+"nrs_" + str(self.ep) + "ep_" + pre
        #number of episodes (and 500-pretrained?) has to be hand-adjusted, perhaps change that...
        return n

def doTraining(parameterset: ParameterSet, number_jobs):
    # Run training loop for multiple iterations (one setting, repeated with the same qtable)
    # If using existing q-table, load from file
    time_s = datetime.now()
    tablefromfile = False
    if tablefromfile:
        # print("Loading Q-Table from file")
        qtableee = QTable(parameterset.LEARNING_RATE, parameterset.DISCOUNT_FACTOR, parameterset.INITIAL_Q_VALUE, parameterset.foldername+parameterset.create_name(), boolchange=parameterset.boolchange)
    else:
        # print("Setting up Q-Table")
        qtableee = QTable(parameterset.LEARNING_RATE, parameterset.DISCOUNT_FACTOR, parameterset.INITIAL_Q_VALUE)
    s_table = np.zeros((4, 4, 4, 2, 2), dtype=np.uint32) #statecount-table 
    # same dimensionality as qtable, but no action-options (because we just want to know about the state... for now)
    # statecount-table (to measure how many times a state was updated)
    rewardcurve = [] #is created by the 
    scorelist = []
    grabslist = []
    tagslist = []
    index = 0 
    for i in range(parameterset.ep): #set batch 6
    #for i in range(500): #set batch 7
    #while datetime.now().hour < 11 or datetime.now().hour > 20: #train until 1 am, then save the q-table and reward curve
        # print("Beginning training run at time ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        seeed = np.random.randint(0, 100000) #random seed while training, set of seeds when testing (TODO)
        #logstructure = []
        timel = 600.
        if index < 500 and parameterset.pretrain: #pretraininng with easy opponents, for more exploration on opponent base  [pretraining"easy" disabled for now, all training against easy(now hard)]
            rewardsteps, capture_entry, grab_entry, tag_entry, u_table = train_qlearn(s_table, seed=seeed, difficulty="easy", reward_choice=parameterset.rewardchoice, render_mode=None, timelimit=timel, q_table=qtableee) #might make timelimit a parameterset choice too...
            qtableee.qtable = u_table.qtable
            # tags, rewardlist, captures, grabs are all for [0] and [1] (the two teams)
            # After each episode update the values of q-table. For this purpose updates are calculated during the episode into the u-table. Afterwards it gets switched with q-table.
        else:
            rewardsteps, capture_entry, grab_entry, tag_entry, u_table = train_qlearn(s_table, seed=seeed, difficulty=parameterset.dif, reward_choice=parameterset.rewardchoice, render_mode=None, timelimit=timel, q_table=qtableee)
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
        if (index % 50) == 49: 
            # print(f"(Pre-storing q-table to file \"{parameterset.foldername + parameterset.create_name()}_q_table.npy\" at index {index}.)")
            qtableee.toFile(f"{parameterset.foldername + parameterset.create_name()}_q_table.npy")
            # print(f"(Pre-storing rewardcurve to file \"{parameterset.foldername + parameterset.create_name()}_reward_curve.npy\" at index {index}.)")
            np.save(f"{parameterset.foldername + parameterset.create_name()}_reward_curve.npy", rewardcurve)
            #print("Storing logstructure to file", "logstructure.npy")
            #np.save(f"{parameterset.foldername + parameterset.create_name()}_logstructure.npy", logstructure) 
            # print(f"(Pre-storing scores to file \"{parameterset.foldername + parameterset.create_name()}_scores.npy\" at index {index}.)")
            np.save(f"{parameterset.foldername + parameterset.create_name()}_scores.npy", scorelist) 
            # print(f"(Pre-storing grabslist to file \"{parameterset.foldername + parameterset.create_name()}_grabslist.npy\" at index {index}.)")
            np.save(f"{parameterset.foldername + parameterset.create_name()}_grabslist.npy", grabslist)
            # print(f"(Pre-storing tagslist to file \"{parameterset.foldername + parameterset.create_name()}_tagslist.npy\" at index {index}.)")
            np.save(f"{parameterset.foldername + parameterset.create_name()}_tagslist.npy", tagslist) 
            # print(f"(Pre-storing statecount-table to file \"{parameterset.foldername + parameterset.create_name()}_statecount.npy\" at index {index}.)")
            np.save(f"{parameterset.foldername + parameterset.create_name()}_statecount.npy", s_table) 
            #print("Statecount table: ",s_table)
        # Print qtable regularly as checkpoint to additional file (but not too oft because memory leak)
        if (index % 500) == 499: 
            # print(f"(In-between-storing q-table to file \"{parameterset.foldername + parameterset.create_name()}_q_table_i{index}.npy\".)")
            qtableee.toFile(f"{parameterset.foldername + parameterset.create_name()}_q_table_i{index}.npy")

        #np.save(f"{filename_suffix}_logstructure{index}.npy", logstructure) 
        # discard logstructure now, so memory does not leak
        #logstructure = []
        index += 1
        # print(f"Completed training run {index}")

    # Epilog (saving q-table and reward curve to file)
    # print(f"Storing q-table to file \"{parameterset.foldername + parameterset.create_name()}_q_table.npy\".")
    qtableee.toFile(f"{parameterset.foldername + parameterset.create_name()}_q_table.npy") #Hmm. Do we need a better naming system, some way to keep track of trained  policies (maybe even in thesis? certainly in slides...), better way to automatically name things, actual pipeline in general
    #testqtable = QTable("q_table.npy")
    # print(f"Storing rewardcurve to file \"{parameterset.foldername + parameterset.create_name()}_reward_curve.npy\".")
    np.save(f"{parameterset.foldername + parameterset.create_name()}_reward_curve.npy", rewardcurve)
    #print("Storing logstructure to file", "logstructure.npy")
    #np.save(f"{parameterset.foldername + parameterset.create_name()}_logstructure.npy", logstructure) 
    # print(f"Storing scorelist to file \"{parameterset.foldername + parameterset.create_name()}_scores.npy\".")
    np.save(f"{parameterset.foldername + parameterset.create_name()}_scores.npy", scorelist)
    # print(f"Storing grabslist to file \"{parameterset.foldername + parameterset.create_name()}_grabslist.npy\".")
    np.save(f"{parameterset.foldername + parameterset.create_name()}_grabslist.npy", grabslist)
    # print(f"Storing tagslist to file \"{parameterset.foldername + parameterset.create_name()}_tagslist.npy\".")
    np.save(f"{parameterset.foldername + parameterset.create_name()}_tagslist.npy", tagslist)
    print(f"Training length: {datetime.now() - time_s} (h:min:sec)", flush=True)
    with lock:
        # counter += 1
        counter.value += 1
        print(f"Concluded experiment {counter.value} out of {number_jobs}", flush=True)
        # print(f"Concluded job {counter} out of {number_jobs}")
#End of doTraining

if __name__ == "__main__":
    timestamp = datetime.now()
    print("Starting experiments at ",timestamp.now().strftime("%d-%m-%Y %H:%M:%S"))
    #if len(sys.argv) > 1:
    # Selecting preset of training parameters ("train" to make sure the files are not overwritten by mistake)
    parametersets = []
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Do large batch training here.
        #########################################
        #vanilla parameters WITHOUT pretrain #batch 6 part one:
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.1, 0.9, 10.0, False, "avgtest1", "qtrainlog/batch 6 avg/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, False, "avgtest1", "qtrainlog/batch 6 avg/", i))
        #vanilla parameters with pretrain
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.1, 0.9, 10.0, True, "avgtest1", "qtrainlog/batch 6 avg/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, True, "avgtest1", "qtrainlog/batch 6 avg/", i))
        
        # 2nd set of parameters without pre #batch 6 part two:  (->18h für 80 trainings)
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.9, 10.0, False, "avgtest2", "qtrainlog/batch 6 part two/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.2, 0.9, 10.0, False, "avgtest2", "qtrainlog/batch 6 part two/", i))
        # # 3rd set of parameters without pre
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.1, 0.95, 10.0, False, "avgtest2", "qtrainlog/batch 6 part two/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.95, 10.0, False, "avgtest2", "qtrainlog/batch 6 part two/", i))
        
        ## 4th set of parameters without pre (will be run tomorrow) #batch 6 part three:    (->26h für 120 trainings)
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.2, 0.95, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", i))
        ## 5th set of parameters without pre 
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.15, 0.9, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.15, 0.9, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", i))
        ## 6th set of parameters without pre 
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.85, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.2, 0.85, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", i))

        # Post-reset (13.4.26, reset to state of 18.3.26 because math2 did not work out)
        # for i in range(20):
        #     parametersets.append(ParameterSet("aggressive_tags", "hard", 0.2, 0.95, 10.0, False, "testrestored", "qtrainlog/batch 10/", i))
        # Changed the boolean b_flag to represent 'on_side'. Testing effect quickly:
        # for i in range(10):
        #     parametersets.append(ParameterSet("aggressive_tags", "hard", 0.2, 0.95, 10.0, False, "testboolchange", "qtrainlog/batch 10/", i))
        # again, it is somewhat promising, but I want to see some wins:
        # for i in range(20):
        #     parametersets.append(ParameterSet("aggressive_tags", "hard", 0.1, 0.9, 10.0, False, "testboolchange2", "qtrainlog/batch 10/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "testboolchange2", "qtrainlog/batch 10/", i))
        # for i in range(20):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "testrestored2", "qtrainlog/batch 10/", i, boolchange=False)) #should this be better documented in the filename? 
        '''Only one run has gotten major winrate so far (standard 0.2 0.95 oldbool math1 that was best before too). I need an extensive test on its conversion rate and need to test newbool thorougly too.'''
        # for i in range(50):       #TODO I need better visualization on old plots to gauge the performance (conversion rate!) of the so far standard setting. I need all single score curves in one plot or simmilar. Only then it makes sense to run a large test again(?).
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "x30restored", "qtrainlog/batch 10/", i, boolchange=False))
        # Testing the performance of boolchange to see if it works or not.
        for i in range(20):
            parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=20, ep=1000)) #see if this improves "chasing" behaviour.
        for i in range(10):
            parametersets.append(ParameterSet("aggressive_tags", "hard", 0.01, 0.95, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
        for i in range(10):
            parametersets.append(ParameterSet("aggressive_tags", "hard", 0.2, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
        for i in range(10):
            parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.01, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
        for i in range(10):
            parametersets.append(ParameterSet("aggressive_tags", "hard", 0.01, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000)) 
        #########################################
        #rewardchoice = "single_aggressive_rew"
        #rewardchoice = "double_aggressive_rew" (outdated)
        #rewardchoice = "caps_and_grabs" (outdated)
        #rewardchoice = "caps_and_tags"
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        print("evaluation pipeline not yet implemented")
        sys.exit(0)
    else:
        # Do visual test match here.
        setup = ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "example", "qtrainlog/example_folder/", 0) #lrate: Any, discount: Any, initialq: Any, pretrain: Any, name: Any, folder)
        #setup = ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", 0) #TODO is this dangerous to overwrite my thing?
        # (do i need to change this so it loads a file?) prolly, 'cause it is for visual test match of trained policy
        rewardchoice = "single_aggressive_rew"
        #filename_suffix = "/vshard_lrate0.1_discount0.95_initialq10.0_single_aggressive_rew"
        print("Manually testing qtable policy with rendering enabled.")
        #qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, (setup.foldername + setup.create_name() + "_q_table.npy"))
        qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/example_folder/avgtest3_single_aggressive_rew_hard_lrate0.2_discount0.95_initq10.0_1000ep_no_pre_nr2_q_table.npy")
        st = np.zeros((4, 4, 4, 2, 2), dtype=np.int32) #dangerous int8-hazard (int8 is insufficient here)
        train_qlearn(st, seed=0, difficulty="hard", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)#TODO change to use parameterset i guess...
        sys.exit(0)

    # Run all scheduled parameters in parallel
    num_jobs = len(parametersets)
    counter.value = 0

    num_workers = 32#20 #15 was best number for my PC in small tests... (cores is 12)
    #num_workers = max(1, os.cpu_count())#TODO test performance of +5 (12 cores, 17 processes now)
    print(f"Selecting {num_workers} as num_workers.")

    with Pool(processes=num_workers) as pool:
        # pool.map(doTraining, parametersets)
        # Need strange partial to fix the num_jobs number into the doTraining calls, just for a counter ("job 4 of 80")
        doTrainingWithCounter = partial(doTraining, number_jobs=num_jobs)
        # Now map the function
        pool.map(doTrainingWithCounter, parametersets)

    # for i in range(len(parametersets)): #NEVER have this uncommented when pool.map is running (I forgot to comment this out :/ )
    #     #run parameterset
    #     doTraining(parametersets[i], num_jobs)
        
        # with lock:
        #     counter.value += 1
        #     print(f"Concluded q-table training {counter} out of {num_jobs}")

    # Print the time the code took
    end_time = datetime.now()
    elapsed_time = end_time - timestamp
    print(f"Total execution time: {elapsed_time} (h:min:sec) for {len(parametersets)} policy trainings.")
    # beware, this line is useless if the terminal is closed without reading the final print. Perhaps print statistics including these to a small textfile?
