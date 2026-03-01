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
from pyquaticus.utils.rewards import caps_and_grabs, aggressive_rew, double_aggressive_rew

"""
This was copied from magnus_test.py and modified to 
create a test of 2x qlearn policy agents versus 
2x base_combined agents during the Masters thesis (Feb 2026).
"""

# To run the training this is called as a function, with MODE and reward function as parameters.

def train_qlearn(
    rewardcurve: str,
    seed: int = 12345,
    # seed for "random" starts
    difficulty: str = "hard",
    # difficulty is the MODE of the example agents, can be "hard", "medium" or "easy"
    # Adjust the mode here to change difficulty of the Heuristic_CTF_Agents
    #MODE = "hard"
    #MODE = "medium"
    #MODE = "easy"
    #agent_type: int = 1,
    # 1 is mrhea agent, 2 is rhea plus two ultra-defensive agents, 3 (or 0) is ultra-defensive agent TODO also add easy,medium,hard as types? maybe not...
    reward_choice: str = "adjustmepls", #maybe could be string, but cleaner so?
    render_mode: str = None,#'human'
    timelimit: float = 600.,
    logname: str = "match.log",
    q_table: str = None
):
    
    # Set score function to the selected reward (match statement syntax might require python version 3.10 or newer)
    match reward_choice:#TODO update reward functions and then this TODO TODO this on sunday!
        case 1:
            reward_method = caps_and_grabs
        case _:
            reward_method = double_aggressive_rew

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

    # initialize rhea environment #not rhea here, at least as long as I don't train against rhea...
    #if agent_type == 1:
    #    rhea_env = MRHEA_Environment(reward_choice)     #(self.id, self.team, obs, info, self.teammate_ids, self.opponent_ids)
    #else:
    #    # rhea and mrhea require different environments (do they rn though?)
    #    rhea_env = RHEA_Environment(reward_choice)
    # give this to the agent below

    # Base_combine agents
    H_one = Heuristic_CTF_Agent('agent_2', env, mode=difficulty, continuous=False)#TODO try if False works (seems more fair)
    H_two = Heuristic_CTF_Agent('agent_3', env, mode=difficulty, continuous=False)
    # Ultra-defensive agents and RHEA agent
    #R_one = UltraDefender('agent_0', rhea_env, env, continuous=True) # MRHEA agent here
    #if agent_type == 1:
    #    print("Setting up MRHEA agent")
    #    R_two = MRHEA_Agent('agent_1', rhea_env, env, continuous=True) # MRHEA agent here
    #elif agent_type == 2:
    #    print("Setting up RHEA agent")
    #    R_one = UltraDefender('agent_0', env, continuous=True) # MRHEA agent here
    #    R_two = RHEA_Agent('agent_1', rhea_env, env, continuous=True)
    #    R_three = UltraDefender('agent_2', env, continuous=True) # MRHEA agent here
    #    #R_one = UltraDefender('agent_0', env, continuous=True) #snippet from rhea_test_cap.py
    #    #R_two = RHEA_Agent('agent_1', rhea_env, env, continuous=True) # RHEA agent here
    #    #R_three = UltraDefender('agent_2', env, continuous=True)
    #    #R_three = UltraDefender('agent_2', rhea_env, env, continuous=True) # MRHEA agent here
    #else:
    #    print("Setting up Ultra-defensive agents")
    #    R_one = UltraDefender('agent_0', env, continuous=True)
    #    R_two = UltraDefender('agent_1', env, continuous=True)
    #    R_three = UltraDefender('agent_2', env, continuous=True)
    #print("Setting up q-table") #is done before training loop
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
        # Because of different agent types we have to distinctively handle the compute_action. This could be avoided with a change to the UltraDefender() method
        #if agent_type == 1:
        #    # MRHEA agents
        #    zot = R_two.compute_action(obs, info)
        #    #print("zot: ",zot) #zot is rn only one action
        #    zero = zot[0]
        #    one = zot[1]
        #    two = zot[2]
        #else:
        #    zero = R_one.compute_action(obs, info)
        #    one = R_two.compute_action(obs, info)
        #    two = R_three.compute_action(obs, info)
        # Only handle qlearn training (for now)
        zero = R_one.compute_action(obs, info)
        one = R_two.compute_action(obs, info)

        # For both agents necessary data for q-value update is saved:
        a0_qstep = R_one.q_Table.prepareUpdate(obs, 'agent_0', zero)
        a1_qstep = R_two.q_Table.prepareUpdate(obs, 'agent_1', one)
        #(ownpos, opp1_bearing, opp2_bearing, b_flag, r_flag, action)
        
        # 2v2 step
        obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three})
        #print("\n\nReward:",reward)#Reward: {'agent_0': 0.2734431070341813, 'agent_1': 0.2208806900959132, 'agent_2': -0.12809429110777018, 'agent_3': 0.0, 'agent_4': 0.0, 'agent_5': 0.0}
        
        # Update Q-Table for both agents (same table, two updates)
        #print("\nActions: zero",zero,"; one",one)
        R_one.q_Table.set_q_value(a0_qstep[0], a0_qstep[1], a0_qstep[2], a0_qstep[3], a0_qstep[4], a0_qstep[5], reward['agent_0'])
        R_two.q_Table.set_q_value(a1_qstep[0], a1_qstep[1], a1_qstep[2], a1_qstep[3], a1_qstep[4], a1_qstep[5], reward['agent_1'])
        # Keep track of reward (TODO need to get an underlying curve and visualize it for full training)
        rewardcurve.append(reward)
        # -Logging utility-
        # Writes the gamestate info into pyquaticus/match.log #this seems like it is doubled? TODO is there another log file output?
        logging.info("obs: %s", obs)
        logging.info("reward: %s", reward)
        logging.info("info: %s", info)

        k =  list(term.keys()) #what is k? idk...
        # In order to keep the simulated environment start state up to date with the "real" one we do the step here (alternative is copying the real one at every step.)
        #if agent_type == 1 or agent_type == 2:
        #    R_two.rhea_env.perform_action({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five}, env.state)

        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            break
    # These are some statistics we are exporting(?):
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

    print("\n~~~Run Concluded~~~")#\nreward curve: ",rewardcurve)
    formatted_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    print(f" Time: {formatted_time}")
    print("agent collisions:",env.state['agent_collisions'])
    print("SCORE: ",env.state['captures'])
    print("grabs: ",env.state['grabs'])
    env.close()
    return rewardcurve

if __name__ == "__main__":
    if False:
        import matplotlib.pyplot as plt
        reward_curve = np.load("reward_curve_aggr_fr_neutral_rew.npy", allow_pickle=True)
        agent_0_rewards = [step['agent_0'] for step in reward_curve]
        agent_1_rewards = [step['agent_1'] for step in reward_curve]
        #print(agent_0_rewards)
        plt.figure(figsize=(12, 6))
        plt.plot(agent_0_rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward Curve")
        plt.grid(True)
        plt.show()
    if False:
        qtablo = QTable("q_table_doubleagg_rew.npy")
        print(qtablo.qtable)
        print("\n\n\n")
        qtablo = QTable("q_table_aggrshort_rew.npy")
        print(qtablo.qtable)
    if True:
        print("Setting up Q-Table")
        qtableee = QTable()
        rewardcurve = []
        index = 0       #right now set for 6h training
        #for i in range(200):
        while datetime.now().hour < 11 or datetime.now().hour > 20: #train until 1 am, then save the q-table and reward curve (TODO visualize the reward cuve later)
            print("Beginning training run at time ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            seeed = np.random.randint(0, 100000) #random seed while training, set of seeds when testing (TODO)
            if index < 500 or False:
                train_qlearn(rewardcurve, seed=seeed, difficulty="easy", reward_choice=2, render_mode=None, timelimit=600., q_table=qtableee)
            elif(index < 200) and False:
                train_qlearn(rewardcurve, seed=seeed, difficulty="medium", reward_choice=2, render_mode=None, timelimit=600., q_table=qtableee)
            else:
                train_qlearn(rewardcurve, seed=seeed, difficulty="hard", reward_choice=2, render_mode=None, timelimit=600., q_table=qtableee)
            index += 1
            print(f"Completed training run {index}")
        #filename = ""
        print("Storing q-table to file", "q_table.npy")
        qtableee.toFile("q_table_aggr_hard_overnight_neutral_rew.npy") #TODO better naming system, some way to keep track of trained  policies (maybe even in thesis? certainly in slides...), better way to automatically name things, actual pipeline in general
        #testqtable = QTable("q_table.npy")
        print("Storing reward curve to file", "reward_curve.npy")
        np.save("reward_curve_aggr_hard_overnight_neutral_rew.npy", rewardcurve)
    if False:
        qtablo = QTable("q_table_aggr_fr_neutral_rew.npy")
        rewardcurve = []
        train_qlearn(rewardcurve, seed=12345, difficulty="hard", reward_choice=2, render_mode='human', timelimit=600., q_table=qtablo)