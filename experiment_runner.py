import os
import numpy as np # type: ignore
import random
#from test.magnus_test import setup_experiment #why does this not work? relative imports are so confusing... <-the answer was (once more) the __init__.py file.
#from magnus_test import setup_experiment
#from .test.magnus_test import setup_experiment
from test.magnus_test import setup_experiment

REPETITIONS = 5

# Paths to the experiment log files
EXPERIMENTS = [
    "test/rhea_test_agr.py",
    "test/rhea_test_agr_easy.py",
    "test/rhea_test_agr_med.py",
    "test/rhea_test_cap.py",
    "test/rhea_test_cap_easy.py",
    "test/rhea_test_cap_med.py",
    "test/ultra_def_test.py",
    "test/ultra_def_test_easy.py",
    "test/ultra_def_test_med.py"
]

RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiments():
    # Generate a fixed set of random seeds for consistent comparison
    seeds = []
    while len(seeds) < REPETITIONS:
        seeds.append(random.randint(1, 10000))
    
    # Difficulties to test
    difficulties = ["easy", "medium", "hard"]
    
    # Agent types to compare
    agent_types = [0, 1]
    
    # Reward choices to test
    reward_choices = [0, 1]
    
    # Nested loops to run experiments
    for agent_type in agent_types:
        print(f"\n{'='*20} AGENT TYPE {agent_type} {'='*20}")
        
        for difficulty in difficulties:
            for seed in seeds:
                for reward_choice in reward_choices:
                    print("label 12")
                    setup_experiment(
                        seed=seed,
                        difficulty=difficulty,
                        agent_type=agent_type,
                        reward_choice=reward_choice,
                        render_mode=None
                    )

# Run the experiments when the script is executed
if __name__ == "__main__":
    print("Beginning experiment runs at time ", os.times())
    run_experiments()
