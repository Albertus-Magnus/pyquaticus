# Pyquaticus
This is a [PettingZoo](https://pettingzoo.farama.org/) environment for maritime Capture the Flag with uncrewed surface vehicles (USVs).

## Motivation
This PettingZoo is a _lightweight_ environment for developing algorithms to play multi-agent Capture-the-Flag with surface vehicle dynamics.
## Example
* Supports standard PettingZoo interface for multi-agent RL
* Pure-Python implementation without many dependencies
```
pip install -e .[torch,ray]
```


# Master Project Documentation Magnus Amann 
## 22.8.2025 – 24.10.2530.9.2025

## Setting up Pyquatics
Followed instructions on readme on github. Had to set up Miniconda.
## Setting up Miniconda
* Miniconda should be installed but „conda: not found“. Reading the Getting Started on it’s website for correct setup…
* Installing again but using the documentation on the site: https://www.anaconda.com/docs/getting-started/miniconda/install#linux-x86
* (it looks like the SHA-256 hash value of the other install download "Miniconda3-latest-Linux-x86_64.sh" was wrong, the new one seems correct)
* I selected the option to automatically initialize conda (reversible by `conda init --reverse $SHELL`). This made the terminal display "(base) " at the beginnning of each prompt line.
* Conda commands are now working.
## Setting up Pyquatics (continued)
* Following the instructions on:
https://github.com/mit-ll-trusted-autonomy/pyquaticus/tree/main?tab=readme-ov-file
* Selected the option to "create the full virtual environment -- with RLLib and PyTorch".
* Created environment at /home/magnus-amann/Master_Project/pyquaticus-main/env-full
* Some error was shown in the output (~~see Anhang 1~~)

* I should now be able to activate the pyquaticus environment with:
conda activate /home/magnus-amann/Master_Project/pyquaticus-main/env-full
* Failed.
* Troubleshooting:
* Retrying './setup-conda-env.sh full'
* On the pyquatics readme a different command to activate the environment was found:
```
conda activate env-full/
```
* Now the environment could be activated.
### Testing the Pyquatics setup
* Continuing with the Basic Tests section of the Pyquatics readme, both tests did not run correctly.
```
python ./test/rand_env_test.py
```
* Traceback (most recent call last):
```
  File "/home/magnus-amann/Master_Project/pyquaticus-main/./test/rand_env_test.py", line 24, in <module>
    import gymnasium as gym
ModuleNotFoundError: No module named 'gymnasium'
```
* Might be related to the error when creating the environment?
* Error suggested missing metadata from git was an issue. Restarted the process after cloning the repository (instead of downloading the zip).
* Now the basic tests work.
## Implementing the naive approach
* Looking at the files in the pyquaticus/test/ folder, heuristic agents seem to be sourced from pyquaticus.base_policies.base_combined.

* In the base_combined.py file the behaviour of this particular agent seems to be sourced from base_defender and base_attacker (indeed this agent is described as a combination of both of these in a readme in this folder).

* The key import here is:
```
import pyquaticus.base_policies.base_defend as defend_policy
```
* Copied the file heuristic_test.py and began adjusting it.
* First change the number of agents to three per team. To change the behaviour to that of Otho-Indep for now we just have to change the mode of a team from "hard" to "nothing". More nuanced changes to the behaviour will require to modify a base_policies file.

* The game already starts with the agents in reasonable defensive positions. 
* Whoever submitted this as entry for the competition was a mad genius to get seccond and third rank by only changing "hard" to "nothing" three times...

