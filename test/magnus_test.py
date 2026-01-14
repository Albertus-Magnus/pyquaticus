from datetime import datetime
import logging
import sys
import os
import os.path
import pyquaticus
import numpy # type: ignore
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.multi_rhea_policy import MRHEA_Agent, MRHEA_Environment
from pyquaticus.base_policies.rhealg_policy2 import RHEA_Agent, RHEA_Environment
from pyquaticus.base_policies.ultra_def_policy import UltraDefender
#from pyquaticus.base_policies.rhealg_policy2 import RHEA_Agent, RHEA_Environment
#from pyquaticus.envs.pyquaticus import Team
from pyquaticus.utils.rewards import triple_aggressive_rew, triple_caps_and_grabs

"""
This was copied from heuristic_test.py and modified to 
create a test of 3x selectable policy agents versus 
3x base_combined agents during the Masters thesis (Jan 2026).
"""

# To run the experiments this was modified to be called as a function, with MODE and reward function as parameters.

def setup_experiment(
    seed: int = 12345,
    # seed for "random" starts
    difficulty: str = "hard",
    # difficulty is the MODE of the example agents, can be "hard", "medium" or "easy"
    # Adjust the mode here to change difficulty of the Heuristic_CTF_Agents
    #MODE = "hard"
    #MODE = "medium"
    #MODE = "easy"
    agent_type: int = 1,
    # 1 is mrhea agent, 2 is rhea plus two ultra-defensive agents, 3 (or 0) is ultra-defensive agent TODO also add easy,medium,hard as types? maybe not...
    reward_choice: int = 1, #maybe could be string, but cleaner so?
    render_mode: str = 'human',
    timelimit: float = 600.,
    logname: str = "match.log"
    # TODO add parameter for team size (how to handle compute_action and other pyqua interfaces cleanly?)
):
    
    # Set score function to the selected reward (match statement syntax might require python version 3.10 or newer)
    match reward_choice:
        case 1:
            reward_method = triple_aggressive_rew
        case _:
            reward_method = triple_caps_and_grabs

    # Adjust the mode here to change difficulty of the Heuristic_CTF_Agents
    #MODE = "hard"
    #MODE = "medium"
    #MODE = "easy"

    config_dict = {}
    config_dict["max_time"] = timelimit#600.0
    config_dict["max_score"] = 100
    config_dict["render_agent_ids"] = True
    config_dict["dynamics"] = ["si", "si", "si", "si", "si", "si"]
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

    env = pyquaticus_v0.PyQuaticusEnv(team_size=3, action_space="continuous", config_dict=config_dict, reward_config={'agent_0': reward_method, 'agent_1': reward_method, 'agent_2': reward_method},
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

    # initialize rhea environment 
    if agent_type == 1:
        rhea_env = MRHEA_Environment(env)     #(self.id, self.team, obs, info, self.teammate_ids, self.opponent_ids)
    else:
        # rhea and mrhea require different environments (do they rn though?)
        rhea_env = RHEA_Environment(env)
    # give this to the agent below

    # Base_combine agents
    H_one = Heuristic_CTF_Agent('agent_3', env, mode=difficulty, continuous=True)
    H_two = Heuristic_CTF_Agent('agent_4', env, mode=difficulty, continuous=True)
    H_three = Heuristic_CTF_Agent('agent_5', env, mode=difficulty, continuous=True)
    # Ultra-defensive agents and RHEA agent
    #R_one = UltraDefender('agent_0', rhea_env, env, continuous=True) # MRHEA agent here
    if agent_type == 1:
        print("Setting up MRHEA agent")
        R_two = MRHEA_Agent('agent_1', rhea_env, env, continuous=True) # MRHEA agent here
    elif agent_type == 2:
        print("Setting up RHEA agent")
        R_one = UltraDefender('agent_0', env, continuous=True) # MRHEA agent here
        R_two = RHEA_Agent('agent_1', rhea_env, env, continuous=True)
        R_three = UltraDefender('agent_2', env, continuous=True) # MRHEA agent here
        #R_one = UltraDefender('agent_0', env, continuous=True) #snippet from rhea_test_cap.py
        #R_two = RHEA_Agent('agent_1', rhea_env, env, continuous=True) # RHEA agent here
        #R_three = UltraDefender('agent_2', env, continuous=True)
        #R_three = UltraDefender('agent_2', rhea_env, env, continuous=True) # MRHEA agent here
    else:
        print("Setting up Ultra-defensive agents")
        R_one = UltraDefender('agent_0', env, continuous=True)
        R_two = UltraDefender('agent_1', env, continuous=True)
        R_three = UltraDefender('agent_2', env, continuous=True)

    step = 0
    rewardcurve = []
    while True:
        # Base_combine agents
        three = H_one.compute_action(obs, info)
        four = H_two.compute_action(obs, info)
        five = H_three.compute_action(obs, info)
        # Because of different agent types we have to distinctively handle the compute_action. This could be avoided with a change to the UltraDefender() method
        if agent_type == 1:
            # MRHEA agents
            zot = R_two.compute_action(obs, info)
            #print("zot: ",zot) #zot is rn only one action
            zero = zot[0]
            one = zot[1]
            two = zot[2]
        else:
            zero = R_one.compute_action(obs, info)
            one = R_two.compute_action(obs, info)
            two = R_three.compute_action(obs, info)

        
        obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five})
        
        rewardcurve.append(reward)
        # -Logging utility-
        # Writes the gamestate info into pyquaticus/match.log
        logging.info("obs: %s", obs)
        logging.info("reward: %s", reward)
        logging.info("info: %s", info)

        k =  list(term.keys())
        # In order to keep the simulated environment start state up to date with the "real" one we do the step here (alternative is copying the real one at every step.)
        if agent_type == 1 or agent_type == 2:
            R_two.rhea_env.perform_action({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three, 'agent_4':four, 'agent_5':five}, env.state)

        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            break
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

if __name__ == "__main__":
    setup_experiment(seed=12345, difficulty="hard", agent_type=1, reward_choice=1, render_mode="human", timelimit=600.)
