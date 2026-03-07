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

# Parameters for Q-Learning:
#LEARNING_RATE = 0.1
#DISCOUNT_FACTOR = 0.9
#INITIAL_Q_VALUE = 10.0 #high initial q-value encourages exploration, low (even negative) encourages exploitation

# Agent IDs for 2v2:
# self: agent_0 or agent_1
# opponents: agent_2 and agent_2

class QTable:
    def __init__(self, LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, filename=None):
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.INITIAL_Q_VALUE = INITIAL_Q_VALUE
        if not filename == None:
            self.qtable = np.load(filename)
            #print(self.qtable)
            print("Q-Table loaded, size",self.qtable.size)
            return
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
        self.qtable = np.full_like(self.qtable, self.INITIAL_Q_VALUE) 
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

    """
    Moved set_q_value to QlearnPolicy(), because updates now require both qtables stored in the agent policy.
    """

    def prepareUpdate(self, obs, agentID, action):
        if obs[agentID]['has_flag']:#TODO run test on correctness of adress of value
            # heading towards objective (positional awareness variable) is towards enemy base normally...
            ownpos = headingToState(obs[agentID]['own_home_bearing']) 
        else:
            # ...and towards own base (or map half) if agent has grabbed the enemy flag
            #print("own_home_bearing =", obs[agentID]['own_home_bearing'])#output of this is "own_home_bearing = 117.06809126991149"
            ownpos = headingToState(obs[agentID]['opponent_home_bearing'])
        # compute opp1 (angle (0-3) between own heading and bearing towards opponent 1)
        opp1_bearing = headingToState(obs[agentID][('opponent_0', 'relative_heading')]) #TODO check if address correct
        # compute opp2
        opp2_bearing = headingToState(obs[agentID][('opponent_1', 'relative_heading')]) #TODO same
        # compute b_flag (bool whether opponent has grabbed the blue flag)
        b_flag = int(obs[agentID][('opponent_0', 'has_flag')] or obs[agentID][('opponent_1', 'has_flag')]) #true if any opponent has your flag
        # compute r_flag
        r_flag = int(obs[agentID]['has_flag']) #is boolean, but integer (0-1) is better for array index
        #translate action from [4, 2, 0, 6] to [0, 1, 2, 3]
        if action == 4:
            action_index = 0
        elif action == 2:
            action_index = 1
        elif action == 0:
            action_index = 2
        elif action == 6:
            action_index = 3
        else:
            print("Error: Action not recognized in prepareUpdate() of QTable.")
            action_index = -1
        return (ownpos, opp1_bearing, opp2_bearing, b_flag, r_flag, action_index)

    def toFile(self, filename): #zB "q_table.npy"
        np.save(filename, self.qtable)
#End of QTable()

class QlearnPolicy(BaseAgentPolicy):
    """Implements a BaseAgentPolicy from pyquaticus, especially the compute_action() method."""
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
        #self.agent_id = agent_id
        #self.env = env
        self.q_Table = q_table
        # copy of q_table (update-table) is used for updates of q-values, while old unmodified q_table is used only for action selection (and max(future action) in the update calculation)
        self.u_Table: QTable = np.copy(q_table)

    @staticmethod #either this line or self as first argument. Python stinks sometimes...
    def headingDiff(heading, pos1, pos2): #TODO check if math correct
        # heading can range from -180 to 180, we want to compute with 360-range values
        heading = heading + 180.0
        #heading += 0.0000001 #add this if float leads to very small negative values...
        #compute angle (relative to coordinate system north) from pos1 to pos2
        angle = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) * 180 / np.pi #TODO check math
        #compute difference between heading and angle
        diff = (heading - angle) % 360
        diff = diff - 180.0 #outside this function the headings/angles are in -180 to 180 range(?)
        return diff
    
    ###############UPDATE##OF##QVALUE################################################
    def set_q_value(self, ownpos, opp1, opp2, b_flag, r_flag, action, reward: float):
        """Adjusts the value of a q-value in the update-table object stored by this agent policy.
        
        :ownpos: is self-position (relative heading towards objective): 0-3 [order of angles todo]
            
        :opp1: is opponent 1 (relative heading towards opp1): 0-3
            
        :opp2: is opponent 2 (relative heading towards opp2): 0-3
            
        :b_flag: is own flag grabbed: Bool
            
        :r_flag: is opponent flag grabbed: Bool
            
        :action: is action space: 0-3

        :reward: is the reward that was found with the selected action

        reward is from frame n+1, the rest of the values are from frame n (action being the action between these frames, so selected in frame n).
        """
        # old_q is taken from the (during this episode) updated q-value
        old_q = self.u_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][action]

        # Calculation of loss etc as per the qlearn algorithm
        opt_future_value = -100000000000. #"." cause reward is continuous
        for i in range(4): #for every possible action do...
            # opt_future_value is taken from the un-updated qtable that is also used for action selection.
            opt_future_value = max(opt_future_value, self.q_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][ i ])
        # Loss function is used to compute new q-value:
        new_q = (1 - self.q_Table.LEARNING_RATE) * old_q + self.q_Table.LEARNING_RATE * (reward + self.q_Table.DISCOUNT_FACTOR * opt_future_value)
        # LEARNING_RATE and other parameters should remain the same between both tables
        #print(f"Updating Q-value for state ({ownpos}, {opp1}, {opp2}, {b_flag}, {r_flag}) and action {action} from {old_q} to {new_q} based on reward {reward} and optimal future value {opt_future_value}.")
        self.u_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][action] = new_q
    #End of set_q_value()

    def compute_action(self, obs, info: dict[str, dict]):
        # Returns an action index (since we are aiming at discrete agent implementation), according to the ACTION_MAP from config:
        # ACTION MAP:
        # [[1.0,  180], [1.0,  135], [1.0,  90], 
        #  [1.0,   45], [1.0,    0], [1.0, -45], 
        #  [1.0,  -90], [1.0, -135], [0.5, 180], 
        #  [0.5,  135], [0.5,   90], [0.5,  45], 
        #  [0.5,    0], [0.5,  -45], [0.5, -90], 
        #  [0.5, -135], [0.0,    0]] #TODO apply this action-id thing to remove the discrete error from my standard pyquaticus runs
        ''' ownpos is the relative heading towards the objective, divided into 4 areas
            pos3  |  pos0
            -----/_\-----   (agent is /_\)
            pos2  |  pos1
        '''
        # To figure out the best reward we need ownpos, opp1, opp2, b_flag, r_flag, action
        # compute ownpos:
        if obs[self.id]['has_flag']:
            # heading towards objective (positional awareness variable) is towards enemy base normally...
            ownpos = headingToState(obs[self.id]['own_home_bearing']) 
        else:
            # ...and towards own base (or map half) if agent has grabbed the enemy flag
            #print("own_home_bearing =",obs[self.id]['own_home_bearing'])#output of this is "own_home_bearing = 117.06809126991149"
            ownpos = headingToState(obs[self.id]['opponent_home_bearing'])
            #TODO this is 2 positional arguments, so not an angle. perhaps position?
            #print above returns 117.06809126991149 wth?!?
        # compute opp1 (angle (0-3) between own heading and bearing towards opponent 1)
        opp1_bearing = headingToState(obs[self.id][('opponent_0', 'relative_heading')]) #TODO check if address correct
        # compute opp2
        opp2_bearing = headingToState(obs[self.id][('opponent_1', 'relative_heading')]) #TODO same
        # compute b_flag (bool whether opponent has grabbed the blue flag)
        b_flag = int(obs[self.id][('opponent_0', 'has_flag')] or obs[self.id][('opponent_1', 'has_flag')]) #true if any opponent has your flag
        # compute r_flag
        r_flag = int(obs[self.id]['has_flag']) #is boolean, but integer (0-1) is better for array index
        # loop through action (range is 0-3)
        #for loop for self.qtable[ownpos][opp1][opp2][b_flag][r_flag][i]
        q_max = -1000000000000
        a_max = -1
        for i in range(4):
            #print(self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag])
            if q_max < self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i]:
                q_max = self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i]
                a_max = i
        #print("Maximum reward",q_max,"expected for action",i,".")
        #return a_max #translate first to pyquaticus action
        #actions = [[1.0, 0], [1.0, 90], [1.0, 180], [1.0, -90]] #(forward, right, backward, left)
        actions = [4, 2, 0, 6] #(same actions, but as discrete indexes for pyquaticus, according to ACTION_MAP)
        return actions[a_max]
    #End of compute_action()
#End of QlearnPolicy()

def headingToState(heading: float): #TODO test (check if heading is already 360 or something pi)
    """
    [Note: first part of functionality went to headingDiff(), now just -180to180 angle to (0-3) state]
    Computes the difference in angle between :heading: and the line between :pos1: and :pos2:.
    Then divides the 360-degree field into four values (0-3) and returns which one the difference is.
            heading
        return3  |  return0
        --------/_\--------   (agent is /_\)
        return2  |  return1
    """
    # heading can range from -180 to 180, we want to compute with 360-range values
    # (and we want to do it explicitely, to ease confusion)
    heading = heading + 180.0
    #divide into four areas and return which one the difference is
    if heading < 90:
        state = 0
    elif heading >= 90 and heading < 180:
        state = 1
    elif heading >= 180 and heading < 270:
        state = 2
    else: #heading >= 270 and heading =< 360:  
        state = 3
    #print("Assigned state {state} for difference {diff} between heading {heading} and angle {angle}.")#print doesnt work with separation to headingDiff()
    return state
#End of headingToState()

    # obs keys: ['opponent_home_bearing', 'opponent_home_distance', 'own_home_bearing', 'own_home_distance', 'wall_0_bearing', 
    # 'wall_0_distance', 'wall_1_bearing', 'wall_1_distance', 'wall_2_bearing', 'wall_2_distance', 'wall_3_bearing', 
    # 'wall_3_distance', 'scrimmage_line_bearing', 'scrimmage_line_distance', 'speed', 'has_flag', 'on_side', 'tagging_cooldown', 
    # 'is_tagged', 'team_score', 'opponent_score', ('teammate_0', 'bearing'), ('teammate_0', 'distance'), 
    # ('teammate_0', 'relative_heading'), ('teammate_0', 'speed'), ('teammate_0', 'has_flag'), ('teammate_0', 'on_side'), 
    # ('teammate_0', 'tagging_cooldown'), ('teammate_0', 'is_tagged'), ('teammate_1', 'bearing'), ('teammate_1', 'distance'), 
    # ('teammate_1', 'relative_heading'), ('teammate_1', 'speed'), ('teammate_1', 'has_flag'), ('teammate_1', 'on_side'), 
    # ('teammate_1', 'tagging_cooldown'), ('teammate_1', 'is_tagged'), ('opponent_0', 'bearing'), ('opponent_0', 'distance'), 
    # ('opponent_0', 'relative_heading'), ('opponent_0', 'speed'), ('opponent_0', 'has_flag'), ('opponent_0', 'on_side'), 
    # ('opponent_0', 'tagging_cooldown'), ('opponent_0', 'is_tagged'), ('opponent_1', 'bearing'), ('opponent_1', 'distance'), 
    # ('opponent_1', 'relative_heading'), ('opponent_1', 'speed'), ('opponent_1', 'has_flag'), ('opponent_1', 'on_side'), 
    # ('opponent_1', 'tagging_cooldown'), ('opponent_1', 'is_tagged'), ('opponent_2', 'bearing'), ('opponent_2', 'distance'), 
    # ('opponent_2', 'relative_heading'), ('opponent_2', 'speed'), ('opponent_2', 'has_flag'), ('opponent_2', 'on_side'), 
    # ('opponent_2', 'tagging_cooldown'), ('opponent_2', 'is_tagged')]

if __name__ == '__main__':
    q_table = QTable()
    #q_table.set_q_value(1,1,1,1,1,1,1)
    print("utility to test the q-learn policy is in qlearn/qlearn_test.py")

    # During (online-)training two agents are using one shared q-table. 
    # When all actions are executed the new reward is entered into (?!) two q-values(?!).

    # Q-table can be saved to file and used for action selection in a policy.
    #np.save("q_table.npy", q_table)