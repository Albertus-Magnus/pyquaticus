import numpy as np
import os
from ray.rllib.policy.policy import Policy
from solution import solution
from pyquaticus.mctf26_config import config_dict_std as mctf_config
# from gen_config import config_dict_competition as mctf_config
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.utils.rewards import caps_and_grabs
s = solution()
mctf_config["sim_speedup_factor"] = 50
mctf_config["max_time"] = 60000.
mctf_config["default_init"] = False
env = CompPyquaticusEnv(action_space="discrete", config_dict=mctf_config, render_mode='human')
term_g = {'agent_0':False,'agent_1':False,'agent_2':False} #this was already set for 3v3, apparently this works with 2v2? (Perhaps there is an alternative win condition to timelimit that we never hit?)
truncated_g = {'agent_0':False,'agent_1':False,'agent_2':False}
term = term_g
trunc = truncated_g
seed = np.random.randint(0, 10000)
print("Seed:", seed)
reset_opts = {'normalize_obs': False, 'normalize_state': False}
obs, info = env.reset(options=reset_opts, seed=seed)
agents = solution()
step = 0
rewardsteps = []
while True:
    # three = agents.compute_action('agent_3', obs, obs, info)
    # four = agents.compute_action('agent_4', obs, obs, info)
    # five = agents.compute_action('agent_5', obs, obs, info)
    three = 16
    four = 16
    five = 16
    zero = agents.compute_action('agent_0', obs, obs, info)
    one = agents.compute_action('agent_1', obs, obs, info)
    two = agents.compute_action('agent_2', obs, obs, info)
    obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five})
    k = list(term.keys()) #Gameover check.
    step += 1
    # if step == 97:
    #     input("Press Enter to continue...")
    if term[k[0]] == True or trunc[k[0]]==True:
        break