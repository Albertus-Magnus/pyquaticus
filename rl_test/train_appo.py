# (C) 2021 Massachusetts Institute of Technology.

# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

# The software/firmware is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause
import argparse
import gymnasium as gym
import numpy as np
import pygame
import ray
#from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
#from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.algorithms.appo import APPOConfig, APPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.rllib.algorithms.appo import APPO
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen
from pyquaticus.config import config_dict_std
import logging

# Patch for tree library to handle mixed str/tuple keys #Start of patch
import tree
_original_map_structure = tree.map_structure
def _patched_map_structure(func, *structures, **kwargs):
    """Patched version that handles mixed key types"""
    # Flatten keeping type information
    def _safe_flatten(s):
        if isinstance(s, dict):
            items = []
            for k, v in s.items():
                if isinstance(v, dict):
                    flattened = _safe_flatten(v)
                    items.extend([(k, k2, vv) for k2, vv in flattened])
                else:
                    items.append(((k,), v))
            return items
        return [((), s)]
    try:
        return _original_map_structure(func, *structures, **kwargs)
    except TypeError as e:
        if "'<' not supported between instances of 'str' and 'tuple'" in str(e):
            # Fallback: process structures without sorting
            if isinstance(structures[0], dict):
                result = {}
                for key in structures[0]:
                    values = tuple(s[key] if isinstance(s, dict) else s for s in structures)
                    result[key] = func(*values)
                return result
            else:
                return func(*structures)
        raise
tree.map_structure = _patched_map_structure
# End of patch

class RandPolicy(Policy):
    """
    Example wrapper for training against a random policy.

    To use a base policy, insantiate it inside a wrapper like this,
    and call it from self.compute_actions

    See policies and policy_mapping_fn for how policies are associated
    with agents
    """
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass

# Wrapper to adapt heuristic/base policies so they receive info_batch TODO change and fix, "Warning: Base policy requires info as well as obs" and perhaps --render needs some api change??
class HeuristicWrapper(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # Expect a constructed base policy instance in config["base_policy"]
        self.base = config.get("base_policy", None)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # Try several call patterns to adapt to different base policy APIs.
        # Ensure info_batch is forwarded if the base policy supports it.
        if self.base is None:
            return [self.action_space.sample() for _ in obs_batch], [], {}

        # 1) Prefer batch-style compute_actions returning (actions, states, infos)
        if hasattr(self.base, "compute_actions"):
            try:
                out = self.base.compute_actions(
                    obs_batch,
                    state_batches,
                    prev_action_batch,
                    prev_reward_batch,
                    info_batch=info_batch,
                    episodes=episodes,
                    **kwargs
                )
                if isinstance(out, tuple):
                    actions = out[0]
                else:
                    actions = out
                return list(actions), [], {}
            except TypeError:
                # Fallthrough to try other call styles below
                pass

        # 2) Try base.act per-observation, passing per-item info if available
        if hasattr(self.base, "act"):
            actions = []
            for i, obs in enumerate(obs_batch):
                info = info_batch[i] if info_batch is not None and i < len(info_batch) else None
                try:
                    # Many heuristics accept (obs, info) or just (obs,)
                    a = self.base.act(obs, info) if info is not None else self.base.act(obs)
                except TypeError:
                    a = self.base.act(obs)
                actions.append(a)
            return actions, [], {}

        # 3) Last resort: sample random actions
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass
# END of HeuristicWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 3v3 policy in a 3v3 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    reward_config = {'agent_0':rew.caps_and_grabs, 'agent_1':rew.caps_and_grabs, 'agent_2':rew.caps_and_grabs, 'agent_3':None, 'agent_4':None, 'agent_5':None} # Example Reward Config
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    RENDER_MODE = 'human' if args.render else None #set to 'human' if you want rendered output
    
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time']=240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob']=True
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode=RENDER_MODE, reward_config=reward_config, team_size=3)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode=RENDER_MODE, reward_config=reward_config, team_size=3))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 'agent_0':
            return "agent-0a-policy"
        if agent_id == 'agent_1':
            return "agent-1a-policy"
        if agent_id == 'agent_2':
            return "agent-2a-policy"
        # Map opposing agents to hard-defend base policies
        if agent_id == 'agent_3':
            return "hard-defend-3"
        if agent_id == 'agent_4':
            return "hard-defend-4"
        if agent_id == 'agent_5':
            return "hard-defend-5"
        return "random"
    
    policies = {
        'agent-0a-policy': (None, obs_space, act_space, {}),
        'agent-1a-policy': (None, obs_space, act_space, {}),
        'agent-2a-policy': (None, obs_space, act_space, {}),
        'random': (RandPolicy, obs_space, act_space, {"no_checkpoint": True}),
        #Examples of Heuristic Opponents in Rllib Training (See two lines below)
        # #'easy-defend-policy': (DefendGen(2, Team.RED_TEAM, 'easy', 2, env.par_env.agent_obs_normalizer), obs_space, act_space, {"no_checkpoint": True})}#,
        #'easy-attack-policy': (AttackGen(3, Team.RED_TEAM, 'easy', 2, env.par_env.agent_obs_normalizer), obs_space, act_space, {})}
        # Hard-defend base policies for opponents (wrapped so info_batch is forwarded)
        'hard-defend-3': (HeuristicWrapper, obs_space, act_space, {"no_checkpoint": True, "base_policy": DefendGen('agent_3', env.par_env, 'hard')}),
        'hard-defend-4': (HeuristicWrapper, obs_space, act_space, {"no_checkpoint": True, "base_policy": DefendGen('agent_4', env.par_env, 'hard')}),
        'hard-defend-5': (HeuristicWrapper, obs_space, act_space, {"no_checkpoint": True, "base_policy": DefendGen('agent_5', env.par_env, 'hard')}),
    }
    env.close()
    #Not using the Alpha Rllib (api_stack False) 
    ppo_config = APPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False).environment(env='pyquaticus').env_runners(num_env_runners=1, num_cpus_per_env_runner=1)
    #If your system allows changing the number of rollouts can significantly reduce training times (num_rollout_workers=15)
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0a-policy", "agent-1a-policy", "agent-2a-policy"],)
    algo = ppo_config.build_algo()
    start = 0
    end = 0
    #for i in range(8001):
    for i in range(360): #ca an hour of training, this is the amount for today's demo
        print("Looping: ", i)
        start = time.time()
        algo.train()
        end = time.time()
        print("End Loop: ", end-start)
        if np.mod(i, 500) == 0:
            print("Saving Checkpoint: ", i)
            chkpt_file = algo.save('./ray_test/iter_'+str(i)+'/')

