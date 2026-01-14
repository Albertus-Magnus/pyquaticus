import os
import numpy as np # type: ignore
import random
from datetime import datetime
from test.magnus_test import setup_experiment

REPETITIONS = 5
MAXTIME = 120.0 #600.0 is standard, currently set for quicker tests

# Create a base results directory with timestamp
def create_experiment_directory():
    # Create a base results directory
    base_results_dir = "experiment_results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Create a timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_results_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir

def run_experiments():
    # Create experiment directory
    experiment_dir = create_experiment_directory()
    #print(f"Experiment results will be saved in: {experiment_dir}")

    # Generate a fixed set of random seeds for consistent comparison
    seeds = []
    while len(seeds) < REPETITIONS:
        seeds.append(random.randint(1, 10000))
    
    # Difficulties to test
    difficulties = ["easy", "medium", "hard"]
    # difficulty of enemy base opponents
    
    # Agent types to compare
    agent_types = [0, 1, 2] 
    #0 is ultra-defensive only, 1 is mrhea, 2 is rhea plus two ultra-defensive agents
    
    # Reward choices to test
    reward_choices = [1, 2]
    #1 is triple_aggressive_rew, 2 is triple_caps_and_grabs
    
    # Loops to run experiments
    for agent_type in agent_types:
        if agent_type == 0 and False:
            continue
        print(f"\n{'='*20} AGENT TYPE {agent_type} {'='*20}")
        
        for difficulty in difficulties:
            for seed in seeds:
                # only iterate through all reward choices for agent_type 1 (mrhea) and 2 (rhea)
                chosen_reward_choices = reward_choices if agent_type == 1 or agent_type == 2 else [reward_choices[0]]
                for reward_choice in chosen_reward_choices:
                    #TODO either the reward function or the compute_action for agent type 1 needs to be fixed, there is an issue.
                    #for reward_choice in reward_choices:
                    print(f"Running agent_type{agent_type}_diff{difficulty}_seed{seed}_reward{reward_choice}")
                    # Create a unique log filename
                    logname = os.path.join(
                        experiment_dir, 
                        f"agent_type{agent_type}_diff{difficulty}_seed{seed}_reward{reward_choice}.log"
                    )
                    #print("label 12")
                    setup_experiment(
                        seed=seed,
                        difficulty=difficulty,
                        agent_type=agent_type,
                        reward_choice=reward_choice,
                        render_mode=None,# "human", # None,# "human", # None,# "human", # None,# "human", # None,# "human", # None,# "human", # None,# "human", # None,# "human", # 
                        timelimit=MAXTIME,
                        logname=logname
                    )

# Run the experiments when the script is executed
if __name__ == "__main__":
    #print("Beginning experiment runs at time ", os.times())
    formatted_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    print(f"~Beginning experiment_runner at time: {formatted_time}~")
    run_experiments()
