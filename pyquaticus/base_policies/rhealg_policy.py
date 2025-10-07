from typing import Any, Union
import numpy as np
from RollingHorizonEA.environment import Environment #if doesnt work add to special import
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
#Start of special rhea import
# try the normal absolute import first; if it fails, add project root and retry
try:
    from RollingHorizonEvolutionaryAlgorithm.RollingHorizonEA.rhea import RollingHorizonEvolutionaryAlgorithm
except ModuleNotFoundError:
    import sys, os, importlib
    # compute project root relative to this file (../../.. -> project root)
    _this_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_this_dir, "..", "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    # retry import (let any exception propagate if it still fails)
    RollingHorizonEvolutionaryAlgorithm = importlib.import_module(
        "RollingHorizonEvolutionaryAlgorithm.RollingHorizonEA.rhea"
    ).RollingHorizonEvolutionaryAlgorithm
# #End of special rhea import

"""
This Class inherits from the Rolling Horizon Evolutionary Algorithm Environment interface.
"""
class RHEA_Environment(Environment):
    
    def __init__(self, agent_id, team, obs, info):
        self.agent_id = agent_id
        self.team = team
        self.obs = obs
        self.info = info

    def perform_action(self, action):
        raise NotImplementedError #TODO

    def evaluate_rollout(self, solution, discount_factor=0, ignore_frames=0):
        raise NotImplementedError #TODO

    def get_random_action(self):
        raise NotImplementedError #TODO

    def is_game_over(self):
        # try: #implement when pyquatic environment is done
        #     return getattr(self._py_env, "game_over", False)
        # except Exception:
        #     return False
        raise NotImplementedError #TODO

    def get_current_score(self):
        raise NotImplementedError #TODO

    def ignore_frame(self):
        raise NotImplementedError #TODO


class RHEA_Agent(BaseAgentPolicy):
    """
    Copied from BaseAgentPolics and modified to implement RHEA agent.
    """

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        suppress_numpy_warnings=True,
        continuous: bool = False,
    ):
        self.id = agent_id
        if self.id in env.agent_ids_of_team[Team.BLUE_TEAM]:
            self.team = Team.BLUE_TEAM
            self.teammate_ids = env.agent_ids_of_team[Team.BLUE_TEAM]
            self.opponent_ids = env.agent_ids_of_team[Team.RED_TEAM]
        elif self.id in env.agent_ids_of_team[Team.RED_TEAM]:
            self.team = Team.RED_TEAM
            self.teammate_ids = env.agent_ids_of_team[Team.RED_TEAM]
            self.opponent_ids = env.agent_ids_of_team[Team.BLUE_TEAM]
        else:
            raise ValueError(f"{self.id} not on a team")
        if suppress_numpy_warnings:
            np.seterr(all="ignore")

        rhea_env = RHEA_Environment(self.id, self.team, obs, info, self.teammate_ids, self.opponent_ids)
        self.rhea = RollingHorizonEvolutionaryAlgorithm(rhea_env)
        # End of __init__

    def compute_action(self, obs, info: dict[str, dict]) -> Any:
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and relative bearing.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """
        action = self.rhea._get_next_action()
        return action
        # End of compute_action
