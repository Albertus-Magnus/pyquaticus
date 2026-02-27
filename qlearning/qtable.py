from typing import Union

from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
#from pyquaticus.utils.rewards import triple_aggressive_rew, triple_caps_and_grabs
import numpy as np
import random
import ast
import re
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.config import config_dict_std, ACTION_MAP

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
INITIAL_Q_VALUE = 1.0 #high initial q-value encourages exploration, low (even negative) encourages exploitation

class QTable:
    def __init__(self):
        #self.statesize = 4^4 # = 256 results in 1024 q-values
        # 2 booleans (for flag grabbed status), 2 headings (four angle-options each) for opp. agents and one self-position variable (four values, either area-based or objective-angle based, see below)
        ''' 
            Action space consists of 4 directions (backwards is effectively similar to zero speed?).
            (speed left, heading right)
            [1.0,    0]
            [1.0,   90]
            [1.0,  180]
            [1.0,  -90]
        '''
        #self.actionsize = 4
        #self.q_table = np.zeros((self.statesize, self.actionsize)) #nope, we want to have it 6-dimensional
        self.qtable = np.zeros((4, 4, 4, 2, 2, 4))
        # Set q-values to initial value (not necessarily zero)
        # initial q-value high encourages exploration, low (even negative) encourages exploitation
        self.qtable = np.full_like(self.qtable, INITIAL_Q_VALUE) 
        ''' This multi-dimensional table stores the qvalues according to the following mapping:
            self-position (relative heading towards objective): 0-3 [order of angles todo]
            opponent 1 (relative heading towards opp1): 0-3
            opponent 2 (relative heading towards opp2): 0-3
            own flag grabbed: Bool
            opponent flag grabbed: Bool
            action space: 0-3
        '''
        print("Q-Table created, size",self.qtable.size)
    #End of init()

    def set_q_value(self, ownpos, opp1, opp2, b_flag, r_flag, action, reward: float):
        """Adjusts the value of a q-value in this QTable object.
        
        :ownpos: is self-position (relative heading towards objective): 0-3 [order of angles todo]
            
        :opp1: is opponent 1 (relative heading towards opp1): 0-3
            
        :opp2: is opponent 2 (relative heading towards opp2): 0-3
            
        :b_flag: is own flag grabbed: Bool
            
        :r_flag: is opponent flag grabbed: Bool
            
        :action: is action space: 0-3

        :reward: is the reward that was found with the selected action

        reward is from frame n+1, the rest of the values are from frame n (action being the action between these frames, so selected in frame n).
        """
        old_q = self.qtable[ownpos][opp1][opp2][b_flag][r_flag][action]

        # Calculation of loss etc as per the qlearn algorithm
        opt_future_value = -100000000000. #"." cause reward is continuous
        for i in range(4): #for every action do...
            opt_future_value = max(opt_future_value, self.qtable[ownpos][opp1][opp2][b_flag][r_flag][ i ])
        # Loss function is used to compute new q-value:
        new_q = (1 - LEARNING_RATE) * old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * opt_future_value)
        #print(f"Updating Q-value for state ({ownpos}, {opp1}, {opp2}, {b_flag}, {r_flag}) and action {action} from {old_q} to {new_q} based on reward {reward} and optimal future value {opt_future_value}.")
        self.qtable[ownpos][opp1][opp2][b_flag][r_flag][action] = new_q
#End of QTable()

class QlearnPolicy(BaseAgentPolicy):
    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        q_table: QTable,
        flag_keepout: float = config_dict_std["flag_keepout"],
        catch_radius: float = config_dict_std["catch_radius"],
        continuous: bool = False
    ):
        super().__init__(agent_id, env)
        self.env = env #is there a reason this isnt in super init? <-prob. not supposed to have this info in-comp.
        self.qtable = q_table

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
    q_table.set_q_value(1,1,1,1,1,1,1)
    # During (online-)training two agents are using one shared q-table. 
    # When all actions are executed the new reward is entered into (?!) two q-values(?!).

    # Q-table can be saved to file and used for action selection in a policy.
    #np.save("q_table.npy", q_table)