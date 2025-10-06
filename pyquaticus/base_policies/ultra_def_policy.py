# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
This was modified from base_defend.py 
to create an ultra-defensive policy
for the Masters project (Oct 2025).
"""

from typing import Any, Union

import numpy as np

from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from utils import local_rect_to_rel_bearing


class UltraDefender(BaseAgentPolicy):
    """
    Ultra-defensie class for avoiding all collisions.
    """

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        continuous: bool = False,
    ):
        super().__init__(agent_id, env)
        self.continuous = continuous
        self.max_speed = env.max_speeds[env.players[self.id].idx]
        self.id = agent_id

    def compute_action(self, obs, info: dict[str, dict]):
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and heading error.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """
        return self.action_from_vector(None, 0)

    def action_from_vector(self, vector, desired_speed_normalized):
        """
        Copied from base_defend.py this is to convert (None, 0) 
        into an action accepted by the Pyquaticus simulation.
        """
        if desired_speed_normalized == 0:
            if self.continuous:
                return (0, 0)
            else:
                return -1
        rel_bearing = local_rect_to_rel_bearing(vector)
        if self.continuous:
            return (desired_speed_normalized * self.max_speed, rel_bearing)
        elif desired_speed_normalized == 0.5:
            if 1 >= rel_bearing >= -1:
                return 12
            elif rel_bearing < -1:
                return 14
            elif rel_bearing > 1:
                return 10
        elif desired_speed_normalized == 1:
            if 1 >= rel_bearing >= -1:
                return 4
            elif rel_bearing < -1:
                return 6
            elif rel_bearing > 1:
                return 2