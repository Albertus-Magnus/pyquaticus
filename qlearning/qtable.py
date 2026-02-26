from typing import Union

from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.rewards import triple_aggressive_rew, triple_caps_and_grabs
import numpy as np
import random
import ast
import re
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.config import config_dict_std, ACTION_MAP

class QTable:
    def __init__(self):
        self.statesize = 4^4 # = 256 results in 1024 q-values
        # 2 booleans (for flag grabbed status), 2 headings (four angle-options each) for opp. agents and one self-position variable (four values, either area-based or objective-angle based, see below)
        ''' 
            Action space consists of 4 directions (backwards is effectively similar to zero speed?).
            (speed left, heading right)
            [1.0,    0]
            [1.0,   90]
            [1.0,  180]
            [1.0,  -90]
        '''
        self.actionsize = 4
        #self.q_table = np.zeros((self.statesize, self.actionsize)) #nope, we want to have it 6-dimensional
        self.qtable = np.zeros((4, 4, 4, 2, 2, self.actionsize))
        print("Q-Table created, size",self.qtable.size)

    #def access_file(self, filename):
        #can be found in qlearn.py

#End of QTable()

class QlearnPolicy(BaseAgentPolicy):
    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        flag_keepout: float = config_dict_std["flag_keepout"],
        catch_radius: float = config_dict_std["catch_radius"],
        continuous: bool = False,
        #mode: str = "easy",
        #defensiveness: float = 20.0,
    ):
        super().__init__(agent_id, env)
        self.env = env #is there a reason this isnt in super init? 
        self.qtable = QPyquaBuilder()

    def compute_action(self, obs, info: dict[str, dict]):
        # Returns an action index (since we are aiming at discrete agent implementation), according to the ACTION_MAP from config:
        # ACTION MAP:
        # [[1.0,  180], [1.0,  135], [1.0,  90], 
        #  [1.0,   45], [1.0,    0], [1.0, -45], 
        #  [1.0,  -90], [1.0, -135], [0.5, 180], 
        #  [0.5,  135], [0.5,   90], [0.5,  45], 
        #  [0.5,    0], [0.5,  -45], [0.5, -90], 
        #  [0.5, -135], [0.0,    0]] #TODO apply this action-id thing to remove the discrete error from my standard pyquaticus runs
        return self.qtable.lookup_action(obs, info)

if __name__ == '__main__':
    q_table = QTable()
    # During (online-)training two agents are using one shared q-table. 
    # When all actions are executed the new reward is entered into (?!) two q-values(?!).

    # Q-table can now be saved and used for action selection in a policy.
    #storing q-table as a numpy file for now, can be loaded in a policy class later:
    np.save("q_table.npy", q_object.q_table)