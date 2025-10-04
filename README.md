# Master Project Documentation Magnus Amann 
## 22.8.2025 – 10.10.25~~30.9.2025~~


## Pyquaticus with added heuristics
This is a [PettingZoo](https://pettingzoo.farama.org/) environment for maritime Capture the Flag with uncrewed surface vehicles (USVs). It contains the [Pyquaticus](https://github.com/mit-ll-trusted-autonomy/pyquaticus) framework as basis and some adjustments, namely an ultra defensive agent policy and an integration of a RHEA implementation.


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
* Attempting to put prints in the rhea.py code (from RollingHorizonEvolutionaryAlgorithm copied folders) made clear that the import didn't work properly.
* Now the import seems to work.
* There is still an error message that is printed with every action selected by an agent:
```
Warning! Action passed in for agent_0 ([  1. -45.]) is not contained in agent's action space (Discrete(17)).
Auto-detecting action space for agent_0
```
* This might be alright, perhaps simply a sign that the framework is autocorrecting the discrete/continuous format of the action. 1 -45 for example seems to be a correct discrete action.
```
All headings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    # This can be converted the standard heading format that is counterclockwise
    # by using the heading_angle_conversion(deg) function found in utils.py
```
* The reward function is not set properly. In pyquaticus.py it is called by:
```self.reward_config[agent_id]```
* Thus it will have to be added on the pyquaticus-environment level.
* Now the reward seems to be defined at the correct place, but a different reward might be necessary to test if it is applied correctly.
* The example reward function does not make sense to me. Are there not two identical values compared to each other?
```
def caps_and_grabs(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    reward = 0.0
    prev_num_oob = state['agent_oob'][agents.index(agent_id)]
    num_oob = state['agent_oob'][agents.index(agent_id)]
    if num_oob > prev_num_oob:
        reward += -1.0
    for t in state['grabs']:
        prev_num_grabs = state['grabs'][t]
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 0.25 if t == team else -0.25

        prev_num_caps = state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 1.0 if t == team else -1.0

    return reward
```
* Right now the rhea thing appears to (maybe?) be working, but planning 300 steps ahead with mostly random moves does not get the agent anywhere close to securing points and thus not close to select an optimal action. Perhaps more branches would help.
* Also the example reward function still does not seem right to me, perhaps something was lost in a past change there in the original repo?
### State of Affairs 5.10.20205
* Whether the example reward function works as intended or not (more likely?), with purely random selection for the rhea probing the agent is not getting anywhere near the necessary position to capture the flag or tag an enemy. This means a much more fluid reward function is needed that always gives a value between 0 and 1. Perhaps the distance to the enemy flag (and with flag to the own base) normalized with the distance between the two flags is a good start. Large distance to enemy agents might be a good secondary...
* Regarding the task from the meeting (26.9.25) about including an evaluation if the achieved reward by the agents increases over time: It seems clear to me that there is not typically an increase over time in the agents behaviour (Lucas et. al. 2016 state that RHEA agents can be "instantly smart with no prior training on a game" and use the term planning more than learning). The term evolution is applied (as far as I can tell) only to the simulated internal model of the game that the agent has. Thus I can observe the reward curve of an agents planning process in a given game-tick/frame to evaluate whether an improvement of the reward takes place. Observing the reward curve of the actual executed actions does not appear to make much sense because it selects the best one it can find from the very start (and would only get an improvement for example when coming in reach of capturing the flag, so not by internal means).
