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
# from pyquaticus.base_policies.multi_rhea_policy import MRHEA_Agent, MRHEA_Environment
# from pyquaticus.base_policies.rhealg_policy2 import RHEA_Agent, RHEA_Environment
# from pyquaticus.base_policies.ultra_def_policy import UltraDefender
from qtable import QlearnPolicy, QTable
# from pyquaticus.utils.rewards import caps_and_grabs, defensive_rew, double_aggressive_rew, single_aggressive_rew, caps_and_tags, aggressive_tags, aggressive_tags_26
from qlearn_test import train_qlearn
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
    def __init__(self, rewardchoice, dif, lrate, discount, initialq, pretrain, name, folder, index, boolchange=True, nrs=20, ep=1000, teamsize3=False, ignoreseed=True, timelimit=600., qtable_suffix=None, sharpturns=True, sim_speedup=3, previous_action=False):
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
        self.teamsize3 = teamsize3
        self.ignoreseed = ignoreseed
        self.timelimit = timelimit
        self.qtablefromfile = qtable_suffix
        self.sharpturns = sharpturns
        self.sim_speedup = sim_speedup
        self.previous_action = previous_action

    # Create a string for file storage that contains all important info about parameters (as well as an index if parameters are used more than once).
    def create_name(self):
        # if self.pretrain:   pre = "500-pretrained" 
        # else:               pre = "no_pre"
        # if self.sharpturns: sh = "sharpturns_" 
        # else:               sh = ""
        # if self.boolchange: nb = "_newbool"
        # else:               nb = ""
        # if self.previous_action: pa = "_prevact"
        # else:               pa = ""
        # the name of all files (qtable, stats, s-table,...):   ("_qtable" etc are appended)
        #n = self.name + "_" + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_" + "1000ep_" + pre + "_nr" + str(self.index) #old name scheme, before nrs and ep were parameterized
        # n = self.name + nb + "_" + pa + "_" + sh + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_"+str(self.nrs)+"nrs_" + str(self.ep) + "ep_" + pre + "_nr" + str(self.index) #now done in _without_index()
        n = self.create_name_without_index() + "_nr" + str(self.index)
        #number of episodes (and 500-pretrained?) has to be hand-adjusted, perhaps change that...
        return n
    
    #TODO boolchange is not printed into the name?
    
    # Create a string for file storage that contains all important info about parameters (as well as an index if parameters are used more than once).
    def create_name_without_index(self):
        if self.pretrain:   pre = "500-pretrained" #NOTE this 500- is missing from the first set of runs (avgtest1?)
        else:               pre = "no_pre"
        if self.sharpturns: sh = "sharpturns_" #has to be included in name, because needs different treatment in the solution.py (or evaluation etc)
        else:               sh = ""
        if self.boolchange: nb = "_newbool"
        else:               nb = ""
        if self.previous_action: pa = "_prevact"
        else:               pa = ""
        # the name of all files (qtable, stats, s-table,...):   ("_qtable" etc are appended)
        #n = self.name + "_" + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_" + "1000ep_" + pre #old name scheme, before nrs and ep (batch 10c and before)
        n = self.name + nb + pa + "_" + sh + str(self.rewardchoice) + "_" + self.dif + "_lrate"+ str(self.LEARNING_RATE) + "_discount" + str(self.DISCOUNT_FACTOR) + "_initq" + str(self.INITIAL_Q_VALUE) + "_" +str(self.nrs)+"nrs_" + str(self.ep) + "ep_" + pre
        #number of episodes (and 500-pretrained?) has to be hand-adjusted, perhaps change that...
        return n

def doTraining(parameterset: ParameterSet, number_jobs):
    #import os, time
    np.random.seed(None)   # reseed from OS entropy
    #(attempt to fix the identical random seeds between parallel executions)

    # Run training loop for multiple iterations (one setting, repeated with the same qtable)
    # If using existing q-table, load from file
    time_s = datetime.now()
    if parameterset.qtablefromfile is not None:
        print("Loading Q-Table from file")
        # qtableee = QTable(parameterset.LEARNING_RATE, parameterset.DISCOUNT_FACTOR, parameterset.INITIAL_Q_VALUE, parameterset.foldername+parameterset.create_name(), boolchange=parameterset.boolchange)
        #stepstest_single_aggressive_rew_nothing_lrate0.1_discount0.9_initq10.0_1nrs_1000ep_no_pre
        # qtableee = QTable(parameterset.LEARNING_RATE, parameterset.DISCOUNT_FACTOR, parameterset.INITIAL_Q_VALUE, 'qtrainlog/batch 3/stepstest_single_aggressive_rew_nothing_lrate0.1_discount0.9_initq10.0_1nrs_1000ep_no_pre_nr0_q_table.npy', boolchange=parameterset.boolchange)
        #enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre
        qtableee = QTable(parameterset.LEARNING_RATE, parameterset.DISCOUNT_FACTOR, parameterset.INITIAL_Q_VALUE, parameterset.qtablefromfile, boolchange=parameterset.boolchange, sharpturns=parameterset.sharpturns, prev_action=parameterset.previous_action)
    else:
        # print("Setting up Q-Table")
        qtableee = QTable(parameterset.LEARNING_RATE, parameterset.DISCOUNT_FACTOR, parameterset.INITIAL_Q_VALUE, boolchange=parameterset.boolchange, sharpturns=parameterset.sharpturns, prev_action=parameterset.previous_action)
    if parameterset.previous_action:
        s_table = np.zeros((4, 4, 4, 4, 2, 2), dtype=np.uint32) #larger statecount-table if more states (due to previous action requiring 4x the q-values)
    else:
        s_table = np.zeros((4, 4, 4, 2, 2), dtype=np.uint32) #statecount-table 
    # same dimensionality as qtable, but no action-options (because we just want to know about the state... for now)
    # statecount-table (to measure how many times a state was updated)
    rewardcurve = [] #is created by the 
    scorelist = []
    grabslist = []
    tagslist = []
    index = 0 
    for i in range(parameterset.ep): #number of episodes is now set by ParameterSet
    #for i in range(500): #set batch 7
    #while datetime.now().hour < 11 or datetime.now().hour > 20: #train until 1 am, then save the q-table and reward curve
        # print("Beginning training run at time ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        seeed = np.random.randint(0, 100000) #random seed while training, set of seeds when testing (TODO) 
        #logstructure = []
        timel = parameterset.timelimit#600.#2000. #600. #TODO think about how much timelimit we should use (right now less because training is longer else)
        # timelimit was computed to be 2000 seconds for 2000 q-updates (and steps), which is the number of updates 24env policies needed to train for
        #NOTE this was <500, should set it back after the extra-large training run...
        if index < 1000 and parameterset.pretrain: #pretraininng with easy opponents, for more exploration on opponent base  [pretraining"easy" disabled for now, all training against easy(now hard)]
            rewardsteps, capture_entry, grab_entry, tag_entry, u_table = train_qlearn(s_table, seed=seeed, difficulty="easy", reward_choice=parameterset.rewardchoice, render_mode=None, timelimit=timel, q_table=qtableee, teamsize3=True, ignoreseed=parameterset.ignoreseed, sim_speed=parameterset.sim_speedup, prev_act=parameterset.previous_action)
            qtableee.qtable = u_table.qtable
            # tags, rewardlist, captures, grabs are all for [0] and [1] (the two teams)
            # After each episode update the values of q-table. For this purpose updates are calculated during the episode into the u-table. Afterwards it gets switched with q-table.
        else:
            rewardsteps, capture_entry, grab_entry, tag_entry, u_table = train_qlearn(s_table, seed=seeed, difficulty=parameterset.dif, reward_choice=parameterset.rewardchoice, render_mode=None, timelimit=timel, q_table=qtableee, teamsize3=True, ignoreseed=parameterset.ignoreseed, sim_speed=parameterset.sim_speedup, prev_act=parameterset.previous_action)
            qtableee.qtable = u_table.qtable

        # Some of the data we are tracking needs to be added to another list structure:
        #rewardcurve.append(rewardsteps)#wrong, need to sum it up exactly before that
        rewardsum = [0., 0.]
        if parameterset.teamsize3:
            rewardsum = [0., 0., 0.]
        #print("\n\n -##########- \grab_entry: ",grab_entry)
        # print(f"Length of episode: 2000sec, {len(rewardsteps)} steps (3x updates to qtable...)")
        for r in rewardsteps:
            # agent0 and agent1 are the ones where reward is interesting for us.
            rewardsum[0] += r['agent_0']
            rewardsum[1] += r['agent_1']
            if parameterset.teamsize3:
                rewardsum[2] += r['agent_2']
        rewardcurve.append(rewardsum) 
        # rewardcurve thus contains agents 0 & 1 as index [0] & [1].
        scorelist.append(capture_entry) #zB [ 0 21]
        grabslist.append(grab_entry) #zB [ 0 23]
        tagslist.append(tag_entry) #zB [ 2 16]

        # Print all important data (especially the q-table!) regularly to file:
        if (index % 100) == 99: 
            # print(f"(Pre-storing q-table to file \"{parameterset.foldername + parameterset.create_name()}_q_table.npy\" at index {index}.)")
            qtableee.toFile(f"{parameterset.foldername + parameterset.create_name()}_q_table_i{index}.npy") #this might be excessive file creation...
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

        #np.save(f"{filename_suffix}_logstructure{index}.npy", logstructure) 
        # discard logstructure now, so memory does not leak
        #logstructure = []
        index += 1
        if index % 100 == 99:
            print(f"Completed training run {index} of {parameterset.create_name()} at time ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"), flush=True)
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
        print(f"Concluded experiment \"{parameterset.create_name()}\" ({counter.value} out of {number_jobs}).", flush=True)
        print(f"latest score: {scorelist[len(scorelist)-1]}")
        # print(f"Concluded job {counter} out of {number_jobs}")
#End of doTraining

if __name__ == "__main__":
    print("-Q-Training experiment commenced.-")
    timestamp = datetime.now()
    print("Starting experiments at ",timestamp.now().strftime("%d-%m-%Y %H:%M:%S"))
    #if len(sys.argv) > 1:
    eval = False
    # Selecting preset of training parameters ("train" to make sure the files are not overwritten by mistake)
    parametersets = []
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("Selecting large batch training.")
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
        #'''Only one run has gotten major winrate so far (standard 0.2 0.95 oldbool math1 that was best before too). I need an extensive test on its conversion rate and need to test newbool thorougly too.'''
        # for i in range(50):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "x30restored", "qtrainlog/batch 10/", i, boolchange=False))
        # Testing the performance of boolchange to see if it works or not.
        # for i in range(20):
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=20, ep=1000)) #see if this improves "chasing" behaviour.
        # for i in range(10):
        #     parametersets.append(ParameterSet("aggressive_tags", "hard", 0.01, 0.95, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
        # for i in range(10):
        #     parametersets.append(ParameterSet("aggressive_tags", "hard", 0.2, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
        # for i in range(10):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.01, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
        # for i in range(10):
        #     parametersets.append(ParameterSet("aggressive_tags", "hard", 0.01, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000)) 
        # batch 1: Testing if 3v3 works in 2026 mctf env. and how promising results are.
        # batch 2: testing against nothing opponent since ultra-defensive is very difficult
        # for i in range(10): #redoing these after bugfix with headings...
        #     parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "3v3test_newbool", "qtrainlog/batch 2/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
        #     #                           mode changed from "hard" to "nothing" for now, because 26env doesnt seem to work with 24-base-policy. Training against ultra-defensive to verify training process...
        # for i in range(10):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.95, 10.0, False, "3v3test_newbool", "qtrainlog/batch 2/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
        # for i in range(10):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.95, 10.0, False, "3v3test_oldbool", "qtrainlog/batch 2/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
        # for i in range(5):
        #     #we have changed these things before this test: reward for caps/grabs 30 to 300, random starting positions, sim_speedup set from 10 to 3.
        #     parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.95, 10.0, False, "speeduptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
        # for i in range(5): #(changed ep to 5 because hpc is not working?)
        #     parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.95, 10.0, False, "speeduptest", "qtrainlog/batch 2/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True))
        # for i in range(5): # this long ep test is using 6000. as max_time. hard agents now work in 26 (only base_attacker so far)
        #     parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True)) 
        # for i in range(2): 
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.1, 0.99, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True)) 
        # for i in range(2):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.85, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
        # for i in range(2):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.1, 0.9, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
        # for i in range(2):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.1, 0.9, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
        # for i in range(2):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.85, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
        # for i in range(2):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.9, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
        # batch 2d
        # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "local-runtime-test", "qtrainlog/batch 2/", 0, boolchange=True, nrs=1, ep=10, teamsize3=True)) 
        # batch 3
        # for i in range(3): #testing if a successful policy results in 2000 sec maxtime (thus 2000 steps and q-updates per episode)
        #     parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.95, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=False, nrs=3, ep=600, teamsize3=True)) 
        # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.9, 10.0, False, "stepstest2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
        # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.99, 10.0, False, "stepstest2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
        # for i in range(3): #stepstest1 did not complete in 8h, now I try stepstest2 and give it a bit of time, as well as 600 (was 1000) episodes.
        #     parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=True, nrs=3, ep=600, teamsize3=True)) 
        # i = 0 #if using i instead of 0 outside of ranges this is necessary. Recommend to just use 0 for clarity
        # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.95, 10.0, False, "stepstest2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
        # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.9, 10.0, False, "stepstest2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
        # batch 3b
        # this is promising? parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.99, 10.0, False, "stepstest2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
        
        #batch 3c
        # nbrs = 20
        # for i in range(nbrs): #Training against moving opponents to avoid circle-deadlocks (that did probably cost us a lot of training time when vs nothing) running a long test to get more robust data on training
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.2, 0.95, 10.0, False, "single26test", "qtrainlog/batch 3/", i, boolchange=False, nrs=nbrs, ep=700, teamsize3=True, timelimit=600.)) 
        # nbrs = 20
        # for i in range(nbrs): #Training against moving opponents to avoid circle-deadlocks (that did probably cost us a lot of training time when vs nothing) running a long test to get more robust data on training
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "single26test", "qtrainlog/batch 3/", i, boolchange=False, nrs=nbrs, ep=700, teamsize3=True, timelimit=600.)) 
        # nbrs = 20
        # for i in range(nbrs): #Training against moving opponents to avoid circle-deadlocks (that did probably cost us a lot of training time when vs nothing) running a long test to get more robust data on training
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.99, 10.0, False, "single26test", "qtrainlog/batch 3/", i, boolchange=True, nrs=nbrs, ep=700, teamsize3=True, timelimit=600.)) 
        #     #TODO if training good reward curve but too short, run further with the qtables from this training run...
        #batch 3d
        # stepstest_single_aggressive_rew_nothing_lrate0.1_discount0.9_initq10.0_1nrs_1000ep_no_pre
        # basically to fix the bug that was present in the grab-return logic of the reward used in this q-policy    #NOTE the first 1000ep are already trained, now 300ep shorter time (600 instead of 2000sec)
        #parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrab", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=300, teamsize3=True)) 
        #once more with 100min episodes
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=300, teamsize3=True)) 
        #nothingslayer1 is created from the 10min enhancegrab qtable
        # parametersets.append(ParameterSet("single_aggressive26", "nothing", 0.1, 0.9, 10.0, False, "nothingslayer1", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=400, teamsize3=True, ignoreseed=True, timelimit=3000.)) 
        # nothingslayer2 uses random starts because non-random starts are too brutal for learning (i.e. result in unhelpful looplike games)
        # parametersets.append(ParameterSet("single_aggressive26", "nothing", 0.1, 0.9, 10.0, False, "nothingslayer2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=500, teamsize3=True, ignoreseed=False, timelimit=3000.)) #parameterset got expanded to include these...
        

        # nbrs = 10
        # for i in range(nbrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "fromzero3000sec", "qtrainlog/batch 4/", i, boolchange=True, nrs=nbrs, ep=1000, teamsize3=True, timelimit=3000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
        
        # TODAY PRIORITY GET PRANAV SOME POLICY FOR COMPETITION EOD!
        # nbrs = 20
        # for i in range(nbrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong4", "qtrainlog/batch 4/", i, boolchange=False, nrs=nbrs, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
        # nbrs = 10
        # for i in range(nbrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong5", "qtrainlog/batch 4/", i, boolchange=True, nrs=nbrs, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
        # nbrs = 5
        # for i in range(nbrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong2", "qtrainlog/batch 3/", i, boolchange=False, nrs=nbrs, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong3", "qtrainlog/batch 3/", i, boolchange=True, nrs=nbrs, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 

        # batch 5a ################ large batch #################
        # nbrs = 10
        # for i in range(nbrs):
        #     # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch1", "qtrainlog/batch 5/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sim_speedup=1)) #before this trained with simspeed 10
        #     # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch2", "qtrainlog/batch 5/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
        #     # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parametersearch3", "qtrainlog/batch 5/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
        #     # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parametersearch4", "qtrainlog/batch 5/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
        #     # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "parametersearch5", "qtrainlog/batch 5/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
        #     #prüfen ob die ersten 60 runs auf dem hpc terminiert haben 
        #     # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "parametersearch6", "qtrainlog/batch 5/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
        #     #ggf müssen die 2 anderen mit höherem sim_speedup getestet werden.
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch7", "qtrainlog/batch 5/", i, boolchange=True, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch8", "qtrainlog/batch 5/", i, boolchange=True, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
        #     #NOTE parametersearch1-8 files are existing in batch5...
        
        #NOTE sharpturns was introduced after this and is a misnamer (if sharpturns is in the name, it is the OLD turning style! ->has to be handled somehow...)
        # # batch 6
        # nbrs = 10
        # for i in range(nbrs): #(wurden nach vorne gelegt: 16 & 17)
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parametersearch17", "qtrainlog/batch 6/", i, boolchange=True, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parametersearch18", "qtrainlog/batch 6/", i, boolchange=True, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nbrs = 10
        # for i in range(nbrs):
        #     # comparison with sharpturns (here non-sharp)
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch9", "qtrainlog/batch 6/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch10", "qtrainlog/batch 6/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parametersearch11", "qtrainlog/batch 6/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parametersearch12", "qtrainlog/batch 6/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "parametersearch13", "qtrainlog/batch 6/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "parametersearch14", "qtrainlog/batch 6/", i, boolchange=False, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch15", "qtrainlog/batch 6/", i, boolchange=True, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch16", "qtrainlog/batch 6/", i, boolchange=True, nrs=nbrs, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            
        # #batch 6b (smaller search on local pc)        (resulted in zero score across the board. Is it even correctly implemented?)
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara1", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=True, sim_speedup=3))
        # parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "shortpara2", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "shortpara3", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # # comparison with sharpturns (here non-sharp)
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara4", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "shortpara5", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "shortpara6", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "shortpara7", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "shortpara8", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "shortpara9", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara10", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "shortpara11", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))

        # batch 6c  (promising parameters training on hpc [and locally, just in case too->6d])
        # nrs = 5
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # # batch 6d
        # nrs = 2
        # for i in range(nrs):                                                                                                          batch 6d <-mistake in setting the correct folder. batch 6d is in 6c folder but recognizable as name contains nrs=2 NOTE
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))

        # batch 6e 2.0  (rerun because bug?)
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6e/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6e/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6e/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6e/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6e/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6e/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))

        # batch 7a (experimenting with previous_action)
        # testing to verify code:
        # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "previoustest0", "qtrainlog/batch 7a/", 0, boolchange=False, nrs=1, ep=10, teamsize3=True, timelimit=60., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        # nrs = 20 #previoustest1 had the most promising results
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "previoustest1", "qtrainlog/batch 7a/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        # nrs = 20 
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "previoustest2", "qtrainlog/batch 7a/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        # nrs = 20 
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "previoustest3", "qtrainlog/batch 7a/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        # nrs = 20 
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "previoustest4", "qtrainlog/batch 7a/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        #tests/debugging (now over):
        # for i in range(3):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "deleteme", "qtrainlog/batch 7a/", i, boolchange=False, nrs=3, ep=3, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        # batch 7b #continuing with the qtables of best results, to see if they profit from longer training
        # nrs = 20 
        # for i in range(nrs):
        #     # nr 8 and 10 were most promising in episodes 901-1000 (1.86 captures on average)
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue1", "qtrainlog/batch 7b/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr10_q_table.npy"))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue2", "qtrainlog/batch 7b/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr8_q_table.npy"))
        # nrs = 20 
        # for i in range(nrs):
        #     # train from qtable (promising one from 1024 size training batch 6e): parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue3", "qtrainlog/batch 7b/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 6e/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy"))
        # # batch 7c longer training time 
        # nrs = 20 
        # for i in range(nrs):
        #     # nr 8 and 10 were most promising in episodes 901-1000 (1.86 captures on average)
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue4", "qtrainlog/batch 7c/", i, boolchange=False, nrs=nrs, ep=4000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))#, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr10_q_table.npy"))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "prevcontinue5", "qtrainlog/batch 7c/", i, boolchange=False, nrs=nrs, ep=4000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))#, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr8_q_table.npy"))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "prevcontinue6", "qtrainlog/batch 7c/", i, boolchange=False, nrs=nrs, ep=4000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 6e/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy"))
        # # batch 8a running pre-training experiment in the modern setting (last time was probably in 24env) and also defensive reward again
        # nrs = 20 
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, True, "pretr1", "qtrainlog/batch 8a/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, True, "pretr2", "qtrainlog/batch 8a/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, True, "pretr3", "qtrainlog/batch 8a/", i, boolchange=True, nrs=20, ep=1000)) 
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.99, 10.0, True, "pretr4", "qtrainlog/batch 8a/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, False, "defender1", "qtrainlog/batch 8a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.99, 10.0, False, "defender2", "qtrainlog/batch 8a/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.15, 0.95, 10.0, False, "defender3", "qtrainlog/batch 8a/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.99, 10.0, False, "defender4", "qtrainlog/batch 8a/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        #     parametersets.append(ParameterSet("caps_and_tags", "hard", 0.15, 0.95, 10.0, False, "defender5", "qtrainlog/batch 8a/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        # # batch 6f (or 6e 3.0)  (rerun because bug?)
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6f/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20                            #TODO timelimit is too high? need to have it at 1200 for best comparability...
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6f/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6f/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6f/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6f/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6f/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #batch 6g (boolchange control group, and sharpturns)
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "boolctrl1", "qtrainlog/batch 6g/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "boolctrl2", "qtrainlog/batch 6g/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6g/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6g/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6g/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6g/", i, boolchange=True, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6g/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=True, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6g/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=True, sim_speedup=3))
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6g/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=True, sim_speedup=3))
        # # batch 6h (getting a confirmation that certain parameters do not result in training success on average; only good parameters were run in breadth so far)
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.85, 10.0, False, "unsuitable_param1", "qtrainlog/batch 6h/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.3, 0.8, 10.0, False, "unsuitable_param2", "qtrainlog/batch 6h/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # batch 6g (aggressive_tags but without newbool as comparison, not sure if i need it for including in thesis but either this or with newbool...)
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.01, 0.99, 10.0, False, "aggrtagsnobool", "qtrainlog/batch 6g/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # # batch 6i (rerunning 6h and 6g2 because single_aggressive_rew was used instead of single_aggressive26 which is not compatible with the env26 and thus can not deliver comparable results)
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.2, 0.85, 10.0, False, "unsuitable_param1", "qtrainlog/batch 6h/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.3, 0.8, 10.0, False, "unsuitable_param2", "qtrainlog/batch 6h/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # nrs = 20
        # for i in range(nrs):
        #     parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "aggrtagsnobool", "qtrainlog/batch 6g/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # batch 7d (continuing with the prevact training to see if this results in a super-qlearning, or is at least able to surpass base-aggr)
        nrs = 20
        for i in range(nrs):
            # pre-training is enabled for half of the attempts. (NOTE pre-training was set to a higher episode limit for this 4k ep run)
            prtr: bool = (nrs > 9)
            parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, prtr, "prevcontinue7", "qtrainlog/batch 7d/", i, boolchange=False, nrs=nrs, ep=4000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix=f"qtrainlog/batch 7c/prevcontinue4_prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_4000ep_no_pre_nr{i}_q_table.npy"))

        #########################################
        #rewardchoice = "single_aggressive_rew"
        #rewardchoice = "double_aggressive_rew" (outdated)
        #rewardchoice = "caps_and_grabs" (outdated)
        #rewardchoice = "caps_and_tags"
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        print("Evaluation pipeline selected.")
        eval = True #marker so q-table will not be modified and data will be output.

        timestamp = datetime.now()
        print("Starting experiments at ",timestamp.now().strftime("%d-%m-%Y %H:%M:%S"))
        
        parametersets = []
        #########################################
        # running quantifying games (50x) for selected parameters
        
        for i in range(50):
            parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.1, 0.9, 10.0, False, "quanttest1", "qtrainlog/eval1/", i))




        #sys.exit(0) #once this works we do not exit here anymore...
    else:
        ####################################### TEST AREA #######################################
        # Do visual test match here.
        setup = ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "example", "qtrainlog/example_folder/", 0, boolchange=True, sharpturns=False) #lrate: Any, discount: Any, initialq: Any, pretrain: Any, name: Any, folder)
        #setup = ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "avgtest3", "qtrainlog/batch 6 part three/", 0) #TODO is this dangerous to overwrite my thing?
        # (do i need to change this so it loads a file?) prolly, 'cause it is for visual test match of trained policy
        rewardchoice = "single_aggressive_rew"
        #filename_suffix = "/vshard_lrate0.1_discount0.95_initialq10.0_single_aggressive_rew"
        print("Manually testing qtable policy with rendering enabled.")
        #qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, (setup.foldername + setup.create_name() + "_q_table.npy"))
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 0/24ver_successful_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 1/3v3test_newbool_aggressive_tags_26_hard_lrate0.01_discount0.95_initq10.0_10nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 1/3v3test_newbool_aggressive_tags_26_hard_lrate0.1_discount0.99_initq10.0_10nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 1/3v3test_oldbool_single_aggressive_rew_hard_lrate0.2_discount0.95_initq10.0_10nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 2/3v3test_oldbool_single_aggressive_rew_nothing_lrate0.2_discount0.95_initq10.0_10nrs_1000ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 2/3v3test_newbool_aggressive_tags_26_nothing_lrate0.1_discount0.99_initq10.0_10nrs_1000ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 2/speeduptest_single_aggressive_rew_nothing_lrate0.2_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 2/speeduptest_aggressive_tags_26_nothing_lrate0.01_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/stepstest2_single_aggressive_rew_nothing_lrate0.1_discount0.9_initq10.0_1nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/stepstest2_aggressive_tags_26_nothing_lrate0.01_discount0.9_initq10.0_1nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/nothingslayer1_single_aggressive26_nothing_lrate0.1_discount0.9_initq10.0_1nrs_400ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/enhancegrablong_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")
        
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/single26test_aggressive_tags_26_hard_lrate0.2_discount0.95_initq10.0_20nrs_700ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/single26test_single_aggressive26_hard_lrate0.15_discount0.99_initq10.0_20nrs_700ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 3/single26test_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_20nrs_700ep_no_pre_nr0_q_table.npy")

        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6b/shortpara6_single_aggressive26_hard_lrate0.01_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6b/shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6b/shortpara10_newbool_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_q_table.npy")
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6c/parameterconfirm18_newbool_aggressive_tags_26_hard_lrate0.15_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_q_table_i899.npy", boolchange=True)
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6c/parameterconfirm18_newbool_aggressive_tags_26_hard_lrate0.15_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_q_table_i949.npy", boolchange=True)
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6c/parameterconfirm18_newbool_aggressive_tags_26_hard_lrate0.15_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_q_table_i999.npy", boolchange=True)
        #                                                                                               Currently running last-minute test to see which qtable I submit to pranav for final comp solution. Not that it mattered, since the competition is much more competetive this year...
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6e/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy", boolchange=False)
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr8_q_table.npy", boolchange=False, prev_action=True, sharpturns=False)
        #testing a 1024 policy (or The policy?) on the 4x statespace, before further traininig for this space:
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6e/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy", boolchange=False, prev_action=True, sharpturns=False)
        ##parametersets.append(ParameterSet("single_aggressive_rew", "hard", 0.3, 0.8, 10.0, False, "unsuitable_param2", "qtrainlog/batch 6h/", i, boolchange=False, nrs=nrs, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6h/unsuitable_param2_single_aggressive_rew_hard_lrate0.3_discount0.8_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy", boolchange=False, prev_action=False, sharpturns=False)
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6f/parameterconfirm17_newbool_aggressive_tags_26_hard_lrate0.01_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy", boolchange=True, prev_action=False, sharpturns=False)
        # op baseline 6f:
        # qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 6f/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy", boolchange=False, prev_action=False, sharpturns=False)
        qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 7c/prevcontinue4_prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_4000ep_no_pre_nr0_q_table.npy", boolchange=False, prev_action=True, sharpturns=False)
    
        # Muster:
        #qt = QTable(setup.LEARNING_RATE, setup.DISCOUNT_FACTOR, setup.INITIAL_Q_VALUE, "qtrainlog/batch 2/_nr0_q_table.npy")
        st = np.zeros((4, 4, 4, 2, 2), dtype=np.int32) #dangerous int8-hazard (int8 is insufficient here)
        seed = np.random.randint(0, 100000)
        print(f"Using seed: {seed}")
        ##(old line:)
        ##rewardsteps, score, grabs, tags, u_table  = train_qlearn(st, seed=seed, difficulty="easy", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt, teamsize3=True, ignoreseed=True, sim_speed=10)
        # Set this True to use the solution policy in opponents/25/solution.py as the opponent
        use_solution_policy = False
        rewardsteps, score, grabs, tags, u_table  = train_qlearn(
            st,
            seed=seed,
            difficulty="easy",
            reward_choice=rewardchoice,
            render_mode='human',
            timelimit=600.,
            q_table=qt,
            teamsize3=True,
            ignoreseed=True,
            sim_speed=10,
            opponent_solution=use_solution_policy,
        )
        print(f"score: {grabs}, grabs: {grabs}")
        timestamp = datetime.now()
        print("Ending experiment at ",timestamp.now().strftime("%d-%m-%Y %H:%M:%S"))
        sys.exit(0)

    # Check if batch folder exists, else create it (checks first parameterset of the queue)
    if not os.path.isdir(parametersets[0].foldername):
        print("Creating folder "+parametersets[0].foldername)
        os.makedirs(parametersets[0].foldername)


    # Run all scheduled parameters in parallel
    num_jobs = len(parametersets)
    counter.value = 0

    if num_jobs <= max(1, os.cpu_count()):
        num_workers = num_jobs
    else:
        num_workers = max(1, os.cpu_count() + 2)
    # or overwrite with own number:
    #num_workers = 60#10#20 #15 was best number for my PC in small tests... (cores is 12)
    #num_workers = max(1, os.cpu_count())#TODO test performance of +5 (12 cores, 17 processes now)
    print(f"Selecting {num_workers} as num_workers.", flush=True)

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
