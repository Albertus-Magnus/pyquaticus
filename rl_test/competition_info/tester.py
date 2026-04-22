import numpy as np
import os
from ray.rllib.policy.policy import Policy
from solution import solution
from pyquaticus.mctf26_config import config_dict_std as mctf_config
# from gen_config import config_dict_competition as mctf_config
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.utils.rewards import caps_and_grabs

s = solution()



env = CompPyquaticusEnv(action_space="discrete", config_dict=mctf_config, render_mode='human')





term_g = {'agent_0':False,'agent_1':False,'agent_2':False} #this was already set for 3v3, apparently this works with 2v2? (Perhaps there is an alternative win condition to timelimit that we never hit?)
truncated_g = {'agent_0':False,'agent_1':False,'agent_2':False}
term = term_g
trunc = truncated_g
#seed = 12345 #SEED for "random" starts
reset_opts = {'normalize_obs': False, 'normalize_state': False}
obs, info = env.reset(options=reset_opts, seed=0)

agents = solution()

# H_one = BaseAttacker('agent_3', env, mode=difficulty, continuous=False)
# H_two = BaseAttacker('agent_4', env, mode=difficulty, continuous=False)
# H_three = BaseAttacker('agent_5', env, mode=difficulty, continuous=False)

# print("Setting up q-learn agents")
# if q_table == None: print("Error: q-table not set up before agents are created.")
# u_table = QTable(q_table.LEARNING_RATE, q_table.DISCOUNT_FACTOR, q_table.INITIAL_Q_VALUE)
# u_table.qtable = np.copy(q_table.qtable)
# q-learn agents
# R_one = QlearnPolicy('agent_0', env, q_table, u_table)
# R_two = QlearnPolicy('agent_1', env, q_table, u_table)
# R_three = QlearnPolicy('agent_2', env, q_table, u_table)


step = 0
rewardsteps = []


while True:
    # Base_combine agents
    three = agents.compute_action('agent_3', obs, obs, info)
    four = agents.compute_action('agent_4', obs, obs, info)
    five = agents.compute_action('agent_5', obs, obs, info)

    zero = agents.compute_action('agent_0', obs, obs, info)
    one = agents.compute_action('agent_1', obs, obs, info)
    two = agents.compute_action('agent_2', obs, obs, info)
    
    
    obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five})
    # print("heading global or not?: ",info['agent_0']['global_state'][('agent_0', "heading")])
    # print("Reward:",reward)#Reward: {'agent_0': 0.2734431070341813, 'agent_1': 0.2208806900959132, 'agent_2': -0.12809429110777018, 'agent_3': 0.0, 'agent_4': 0.0, 'agent_5': 0.0}
    # print(reward["agent_0"],reward["agent_1"],reward["agent_2"])
























# # normalized obs not used by Q-table-based solution, but pass an empty placeholder
# full_obs_normalized = {}
# global_state = {}

# action = s.compute_action("agent_0", full_obs_normalized, full_obs, global_state)
# print("action:", action)