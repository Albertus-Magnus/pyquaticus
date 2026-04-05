import sys

import numpy as np
from numpy.typing import NDArray

from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.mctf26_config import config_dict_std as mctf_config

from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.utils.rewards import caps_and_grabs, aggressive_tags, caps_and_tags, single_aggressive_rew
# from pyquaticus.qlearning.qtable import QTable
# from pyquaticus.base_policies.qlearn_policy import QlearnPolicy
from qlearning.qtable import QlearnPolicy, QTable
# from pyquaticus.qlearning.qtable import QlearnPolicy, QTable  # WHY ARE PYTHON IMPORTS LIKE THIS? one of these should work, no??
# from ..qlearning.qtable import QlearnPolicy, QTable
# from qtable import QlearnPolicy, QTable


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
    q_table: QTable = None #str = None #Not a string?! Is already a QTable!
):
    
    # Set score function to the selected reward (match statement syntax might require python version 3.10 or newer)
    match reward_choice:
        case "caps_and_grabs":
            reward_method = caps_and_grabs
        case "aggressive_tags":
            reward_method = aggressive_tags
        case "single_aggressive_rew":
            reward_method = single_aggressive_rew
        # case "defensive_rew":
        #     reward_method = defensive_rew
        case "caps_and_tags":
            reward_method = caps_and_tags
        # case "aggr_rew_alt":
        #     reward_method = aggr_rew_alt
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

    # env = pyquaticus_v0.PyQuaticusEnv(team_size=2, action_space="discrete", config_dict=config_dict, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method},
    # render_mode=render_mode) #'human')  #None)#'human')
    # env = CompPyquaticusEnv(team_size=2, action_space="discrete", config_dict=config_dict, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method, 'agent_3': reward_method},
    # render_mode=render_mode) #'human')  #None)#'human')
    env = CompPyquaticusEnv(render_mode='human', config_dict=mctf_config) #config_dict)
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
    #u_table = QTable(q_table.LEARNING_RATE, q_table.DISCOUNT_FACTOR, q_table.INITIAL_Q_VALUE)
    #u_table.qtable = np.copy(q_table.qtable)
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
            break

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
    return rewardsteps, env.state['captures'], env.state['grabs'], env.state['tags']#, u_table 
#End of train_qlearn()



if __name__ == "__main__":
    filename_suffix = "batch 1/testing_26env"
    LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.95, 10.0
    # print("Manually testing qtable policy with rendering enabled.")
    qt = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, ("qtrainlog/" + filename_suffix + "_q_table.npy"))
    st = np.zeros((4, 4, 4, 2, 2), dtype=np.int32)
    # train_qlearn(st, seed=0, difficulty="hard", reward_choice=rewardchoice, render_mode='human', timelimit=600., q_table=qt)
    rewardchoice = "aggressive_tags"
    train_qlearn(st, seed=0, difficulty="hard", reward_choice=rewardchoice, render_mode=None, timelimit=600., q_table=qt)
    sys.exit(0)