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
## 22.8.2025 – 10.10.25~~30.9.2025~~



https://github.com/user-attachments/assets/973542d5-bfdf-4421-a0fc-4254b58026bb



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

## The search for the Rolling Horizon Evolutionary Algorithm (RHEA) or the MRHE
* Paper is cited by the main paper: Rolling Horizon Evolutionary Algorithms for General Video Game Playing
* This paper mentions a github repo: https://github.com/rdgain/ExperimentData/tree/NTBEA-RHEA-2019
* This repo contains only stats data and a link to another repo: https://github.com/Bam4d/NTBEA
* This repo contains the NTBEA algorithm which is not the RHE or MRHE but an algorithm that is used to optimize the parameters of RHE.
* Chris Bamford (Bam4d) who is the owner of that repository also has other repositories on github:
```
Older and not well documented:
https://github.com/Bam4d/ALE-Rolling-Horizon
A visualization tool, perhaps useful at a different stage but not RHEA:
https://github.com/rdgain/VERTIGO
Promising, seemingly well documented RHEA:
https://github.com/Bam4d/RollingHorizonEvolutionaryAlgorithm
```
* No MRHE yet found. To use RHEA on a single-member team it has to be integrated into the pyquaticus setup.
### Integrating RollingHorizonEvolutionaryAlgorithm into pyquaticus
* New policy has to be created.
* Parameters and initialization have to be figured out.
* First trying to get a single agent RHEA running in pyquaticus framework.

![plan_tuesday](https://github.com/user-attachments/assets/c81034a8-84bd-4990-b6cf-d6e455de826e)

Currently the rewards or evaluation function is too meaningless. Either the agents need to be better at finding the route towards the next point (capturing a flag) via tweaking of parameters like mutation chance or a better reward needs to be implemented.

<img width="1920" height="1080" alt="Screenshot from 2025-09-25 14-02-20" src="https://github.com/user-attachments/assets/53044600-bfc7-49e9-9e87-0bf9c3d1e1d5" />

* Testing with different return and print labels showed that a lot of times when deepcopy is used it was used on None object instead of pyquaticus environment object.
* set_start_state() exists but was never called, the _start_env variable thus remains None.
* TypeError: cannot pickle 'pygame.surface.Surface' object
* It seems the deepcopy of the environment is now failing to copy an environment due to the game grafics being included in it. We have to copy/wrap the environment without the grafics.
* Some tinkering later this is not resolved. But when changing render_mode=None at the very first instance of env the program runs without this issue. Can the render mode be changed after the agents are created? Would that help?
* Without grafics it runs, but changing the render mode after initialization requires further changes.
* All attempts to remove render mode and reapply it shortly before and after a deepcopy are not working so far.
* There needs to be a different method to copy or a major change to the render mode initialization...
* What if when initializing an agent we initiate a new env in render mode None and write a function to set all parameters so it will be a copy of original env in all other aspects?
* Implemented that, now it runs without deepcopy picklable errors. But the agents are not moving now. This is progress! :D
* There seems to be wrong interaction between agent and env in the sense of different values for the continuous/discrete option. I thought continuous was correct but perhaps discrete works?
* Reading up PyQuaticusEnv readme on the topic:
```
action_space: type of action space for each agent ('discrete', 'continuous', or 'afp')
        (1) 'discrete': discrete action space with all combinations of max speed, half speed; and 45 degree relative heading intervals
        (2) 'continuous': continuous action space for speed [0, max speed] and desired relative heading [0, 359]
        (3) 'afp': discrete action space of target positions from AQUATICUS_FIELD_POINTS (see config.py)

        Note 1: If different action spaces are desired for different agents, provide a list / tuple / array of length 2*team_size like:
                ['discrete', 'discrete', 'continuous', 'afp']
                Each action space type will be applied to the agent at the corresponding index in self.agents.

        Note 2: All agents can take each type of action input to the step function regardless of the type action space specified.
                This parameter is just used to set the PettingZoo standard action_spaces attribute.

        Note 3: Inputs to the step function for the 'afp' action space will be strings AQUATICUS_FIELD_POINTS (see config.py)
```
* Actually seems like discrete vs continuous should not be an issue.
* But other problem found: rhea action is not yet converted to pyquaticus action space. This needs some work.
* Also the issue why the agents were not moving was that the rhea parameters were very large search spaces. So they would've moved when tested for far longer.
