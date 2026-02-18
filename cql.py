from ray.rllib.algorithms.cql import CQLConfig
from ray.tune import CLIReporter
import ray
import argparse
import gymnasium as gym
import numpy as np
import pygame
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec, Policy
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen
from pyquaticus.config import config_dict_std
import logging
# (a lot of unnecessary imports, but better than searching one - copypasted from train_3v3.py)

def train_cql():
    """Train a CQL agent on the pyquaticus game."""
    ray.init()
    #'''
    reward_config = {'agent_0':rew.caps_and_grabs, 'agent_1':rew.caps_and_grabs, 'agent_2':rew.caps_and_grabs, 'agent_3':None, 'agent_4':None, 'agent_5':None} # Example Reward Config
    RENDER_MODE = 'human' #if args.render else None #set to 'human' if you want rendered output
    
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time']=20#240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob']=True


    #   Running this file currently throws an error:
    #   AttributeError: 'Discrete' object has no attribute 'low'), taking actor 1 out of service.
    #   Is this because the action space of the environment is currently a Discrete space, which is not compatible with the CQL implementation?

    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode=RENDER_MODE, reward_config=reward_config, team_size=3)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode=RENDER_MODE, reward_config=reward_config, team_size=3))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    #from ray.rllib.algorithms.cql import CQLConfig
    config = CQLConfig().training(gamma=0.9, lr=0.01)
    config = config.resources(num_gpus=0)
    config = config.env_runners(num_env_runners=4)
    config = config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)
    print(config.to_dict())
    # Build an Algorithm object from the config and run 1 training iteration.
    algo = config.build(env="pyquaticus")
    algo.train()
    #'''
    '''config = (
        CQLConfig()
        .environment("pyquaticus")
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training(
            lr=1e-4,
            train_batch_size=256,
        )
        .evaluation(eval_num_workers=0)
    )'''
    
    #algo = config.build()
    
    #for _ in range(10):
    #    result = algo.train()
    #    print(f"Episode Reward Mean: {result['episode_reward_mean']}")
    
    #ray.shutdown()


if __name__ == "__main__":
    train_cql()