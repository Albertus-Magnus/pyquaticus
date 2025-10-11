# Documentation log Masters Project Magnus Amann

## Setting up Pyquaticus

- This process assumes a linux device (tested with Ubuntu 22.04).
- Clone the repository using git.
- Move to top folder of repo hirachy.
- Miniconda or Anaconda must be installed.
- A virtual environment is created. The option including RLLib and PyTorch was tested:
```
./setup-conda-env.sh full
```
- Afterwards the environment can be activated using ``conda activate env-full/``.
- Within the environment Pyquaticus can be executed, for example: ``python ./test/base_policy_test.py``
- Some dependencies may need to be installed using pip.

## Ultra-Defensive Strategy

- Executing the Pyquaticus environment is done in the ./test/ fodler (starting in the repository root folder). For example ``./test/ultra_def_test.py``.
- base_policy_test.py is calling ``./pyquaticus/base_policies/base_combined.py`` for the agent behaviour (policy).
- To implement the ultra-defensive strategy a ``./test/base_policy_test.py`` copy has to be created and modified to either call a custom base_policy or call an existing policy that supports the argument ``mode="nothing"`` in the python file.
- A custom policy would either have to return ``(0, 0)`` (for continuous agents) or ``Null`` (for discrete agents) when its ``compute_action()`` function is called to implement this behaviour. ``base_policies/ultra_def_policy.py`` was created during this project and implements this strategy.

## Looking for RHEA implementations on GitHub

- RHEA (or rather MRHE) was used by the Lucas-QMU team in the MCTF 2024 competition.
- In "The First International Maritime Capture the Flag Competition: Lessons Learned and Future Directions" one paper is quoted in relation with their approach: "Rolling horizon evolutionary algorithms for general video game playing" (2020; published in IEEE Transactions on Games in 2021). 
- But the focus of this paper is the N-Tuple Bandit Evolutionary Algorithm, which is a method of finding suitable parameters for executing RHEA on a given game.
- RHEA implementations do not appear to be linked or described in detail in this paper.
- Another paper by a similar team (Simon Lucas and Diego Perez-Liebana appear in both) is "Rolling Horizon Coevolutionary Planning for Two-Player Video Games". This paper describes the adaptation of RHEA from one-player video games to two-player video games. This is more essential for the context of Pyquaticus (with two teams and multiple agents per team).
- Under the same account as the GitHub repository for the N-Tuple Bandit Evolutionary Algorithm from above a repository called "RollingHorizonEvolutionaryAlgorithm" can be found. This appears to be the most well-documented version of RHEA by the developers behind the Lucas-QMU approach available on GitHub.
```https://github.com/Bam4d/RollingHorizonEvolutionaryAlgorithm/```
- An older and less documented appearing version by the same account can also be found, but the above version is used in this project.
```
https://github.com/Bam4d/ALE-Rolling-Horizon
```

## Installation of the Rolling Horizon Evolutionary Algorithm

- To use this RHEA implementation in Pyquaticus it is installed within the conda environment. If necessary activate the environment again: ``conda activate env-full/``.
```
pip install RollingHorizonEA
```
- In our case a local copy of the RollingHorizonEvolutionaryAlgorithm repository was used to be installed with pip.

## Implementing RHEA as a Pyquaticus agent policy

- Since the selected RHEA implementation is for a single agent, integration was initially done with a modified Pyquaticus setup with just one agent in each team.
- The core of this implementation is the python class ``RollingHorizonEvolutionaryAlgorithm``. Instances of this class have to be created by a Pyquaticus policy in order to utilize this heuristic for ``compute_action()``.
- Additional implementations of methods accessed in ``RollingHorizonEvolutionaryAlgorithm`` have to be created to adapt this class to the Pyquaticus simulation.
- [Some points about game wrapper, reward/score function and action format may be added here.]

## Running the simulations using the created repository

- Follow the steps in "Setting up Pyquaticus" (above).
- Make sure the conda environment is activated.
- Follow the steps in "Installation of the Rolling Horizon Evolutionary Algorithm" (above).
- The simulations can be executed by using either of these commands:

For a test run of a pre-existing policy that comes with the Pyquaticus framework:
```
python ./test/base_policy_test.py
```

For a run of the ultra-defensive policy against a team using pre-existing Pyquaticus policies:
```
python ./test/nothing_test.py
```

For an implementation of integrating RHEA into the Pyquaticus framework:
```
python ./test/copy_rhea_test.py
```

Depending on the purpose of the execution the render mode has to be adjusted to 'human' or None. This is always done in the file that has "test" in its name. Files that are not the test executers may have different render modes that must not be changed to 'human'.
Example of a human render mode (from ``mrhea_test_agr.py``, ca line 30):
```
env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict, reward_config={'agent_1': triple_aggressive_rew}, render_mode='human')
```
