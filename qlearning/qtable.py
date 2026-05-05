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

# B_FLAG_DELIVERY_POS = np.array([10., 70.]) # (upper left corner)
# R_FLAG_DELIVERY_POS = np.array([150., 70.]) # (upper right corner)

# currently training is only on blue position
FLAG_DELIVERY_POS = np.array([10., 70.]) # (upper left corner)

# the flag in the 26env has to be delivered to one of the two bases in the corners of the own side, not just to the own side anymore.

class QTable:
    def __init__(self, LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, filename=None, boolchange=True, sharpturns=True, previous_action=False):
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.INITIAL_Q_VALUE = INITIAL_Q_VALUE
        self.boolchange = boolchange
        self.sharpturns = sharpturns #decides if 90° turns are the left/right action, if false its 45° turns
        self.prev_action = previous_action #if true, the previous action is added as a parameter to the q-table (increasing its statespace by factor 4).
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
        if self.prev_action:
            self.qtable = np.zeros((4, 4, 4, 2, 2, 4, 4)) #ownpos, opp1, opp2, b_flag, r_flag, action, previous action
        else:
            self.qtable = np.zeros((4, 4, 4, 2, 2, 4))
        # Set q-values to initial value (not necessarily zero)
        # initial q-value high encourages exploration, low (even negative) encourages exploitation
        self.qtable = np.full_like(self.qtable, self.INITIAL_Q_VALUE) 
        ''' This multi-dimensional table stores the qvalues according to the following mapping:
            self-position (relative heading towards objective): 0-3
            opponent 1 (relative heading towards opp1): 0-3
            opponent 2 (relative heading towards opp2): 0-3
            own flag grabbed: Bool
            opponent flag grabbed: Bool
            action space: 0-3
        '''
        # print("Q-Table created, size",self.qtable.size)
    #End of init()

    """
    Moved set_q_value to QlearnPolicy(), because updates now require both qtables stored in the agent policy.
    """

    def prepareUpdate(self, obs, info, agentID, action):
        #delete below (test)
        # if agentID == 'agent_0' and False:
        #     vec_from_agent = np.array([5. - info['agent_0']['global_state'][(agentID, 'pos')][0] , 75. - info['agent_0']['global_state'][(agentID, 'pos')][1] ])
        #     agent_heading = info['agent_0']['global_state'][(agentID, "heading")]
        #     ownpos = headingToState(angle180(vec_to_heading(vec_from_agent) - agent_heading))
        #     print(f"ownpos: {ownpos} heading: {angle180(vec_to_heading(vec_from_agent) - agent_heading)}")
        #####remove above
        if obs[agentID]['has_flag']:            
            # heading towards objective (positional awareness variable) is towards enemy base normally...
            #ownpos = headingToState(obs[agentID]['own_home_bearing']) 
            #ownpos = headingToState(np.array([5.0, 75.0])) #adjusted for 2026 twobase rules
            #ownpos = headingToState(obs[agentID]['wall_0_bearing'])
            #ownpos = headingToState(vec_to_heading(np.array([5., 75.]))) 
            # vec_from_agent = np.array([5. - info['agent_0']['global_state'][(agentID, 'pos')][0] , 75. - info['agent_0']['global_state'][(agentID, 'pos')][1] ])
            agent_heading = info['agent_0']['global_state'][(agentID, "heading")]
            agent_position = info[agentID]['global_state'][(agentID, 'pos')]

            ownpos = headingToState( bearing_from_coord(FLAG_DELIVERY_POS, agent_position, agent_heading) )

            #ownpos = headingToState(vec_to_heading(np.array([5., 75.]))) #TODO test this (then add to prepareUpdate()...)
            # ownpos = headingToState(angle180(vec_to_heading(vec_from_agent) - agent_heading)) #TODO test this (then add to prepareUpdate()...)
            # print(f"ownpos: {ownpos} heading: {angle180(vec_to_heading(vec_from_agent) - agent_heading)}")
        else:
            # ...and towards own base (or map half) if agent has grabbed the enemy flag
            #print("own_home_bearing =", obs[agentID]['own_home_bearing'])#output of this is "own_home_bearing = 117.06809126991149"
            ownpos = headingToState(obs[agentID]['opponent_home_bearing'])
        # compute opp1 (angle (0-3) between own heading and bearing towards opponent 1)
        if len(obs) >= 5: #3v3 #TODO check if this works
            #closest two, not just any two opponents
            if obs[agentID][('opponent_0', 'distance')] < obs[agentID][('opponent_1', 'distance')]:
                #opp0 is one of two closest opp. agents (out of three)
                opp1_bearing = headingToState(obs[agentID][('opponent_0', 'relative_heading')]) 
            else:
                #opp1 is one of two closest opp. agents
                opp1_bearing = headingToState(obs[agentID][('opponent_1', 'relative_heading')]) 
            if obs[agentID][('opponent_1', 'distance')] < obs[agentID][('opponent_2', 'distance')]:
                #opp1 is also one of two closest opp. agents
                opp2_bearing = headingToState(obs[agentID][('opponent_1', 'relative_heading')])
            else:
                #opp2 is also one of two closest opp. agents
                opp2_bearing = headingToState(obs[agentID][('opponent_2', 'relative_heading')])
            # (All closest agents should be sorted out now with the minimal number of comparisons.)
        else: #2v2
            # compute opp1 (angle (0-3) between own heading and bearing towards opponent 1)
            opp1_bearing = headingToState(obs[agentID][('opponent_0', 'relative_heading')])
            # compute opp2
            opp2_bearing = headingToState(obs[agentID][('opponent_1', 'relative_heading')])
        # compute b_flag (bool whether opponent has grabbed the blue flag)
        if self.boolchange:
            b_flag = int(obs[agentID]["on_side"]) #now b_flag represents if opponents are tag-able
        else: 
            if len(obs) >= 5: #TODO check if this works
            #if 3 opponents exist
                b_flag = int(obs[agentID][('opponent_0', 'has_flag')] or obs[agentID][('opponent_1', 'has_flag')] or obs[agentID][('opponent_2', 'has_flag')]) #true if any opponent has your flag Was changed to show on which side of the map we are (as parameter, but so far does not look better).
            else:
                b_flag = int(obs[agentID][('opponent_0', 'has_flag')] or obs[agentID][('opponent_1', 'has_flag')]) #true if any opponent has your flag Was changed to show on which side of the map we are.
        # compute r_flag
        r_flag = int(obs[agentID]['has_flag']) #is boolean, but integer (0-1) is better for array index
        if self.sharpturns:
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
        else:
            #translate action from [4, 3, 0, 5] to [0, 1, 2, 3]
            if action == 4:
                action_index = 0
            elif action == 3:
                action_index = 1
            elif action == 0:
                action_index = 2
            elif action == 5:
                action_index = 3
            else:
                print("Error: Action not recognized in prepareUpdate() of QTable.")
                action_index = -1
        return (ownpos, opp1_bearing, opp2_bearing, b_flag, r_flag, action_index)

    def toFile(self, filename): #zB "q_table.npy"
        np.save(filename, self.qtable)
#End of QTable()

def angle180(deg):
    """Rotates an angle to be between -180 and +180 degrees."""
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg

from pyquaticus.utils.utils import heading_angle_conversion
def vec_to_heading(vec):
    """Converts a vector to a magnitude and heading (deg)."""
    import math
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    return angle180(heading_angle_conversion(angle))

def bearing_from_coord(goal_pos, agent_pos, agent_heading):
    """Computes the relative bearing for a given agent towards a global position/coordinate.
    agent_heading is the current direction (global) the agent is directed/headed/turned 
    towards, in degrees between -180 and 180.
    for example: info['agent_0']['global_state'][(agentID, "heading")] """
    vec_from_agent = np.array([goal_pos[0] - agent_pos[0] , goal_pos[1] - agent_pos[1]])
    return angle180(vec_to_heading(vec_from_agent) - agent_heading)

class QlearnPolicy(BaseAgentPolicy):
    """Implements a BaseAgentPolicy from pyquaticus, especially the compute_action() method."""
    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        q_table: QTable,
        u_table: QTable,
        flag_keepout: float = config_dict_std["flag_keepout"],
        catch_radius: float = config_dict_std["catch_radius"],
        continuous: bool = False
    ):
        super().__init__(agent_id, env)
        #self.agent_id = agent_id
        #self.env = env
        self.q_Table = q_table
        # copy of q_table (update-table) is used for updates of q-values, while old unmodified q_table is used only for action selection (and max(future action) in the update calculation)
        self.u_Table = u_table

        if self.q_Table.prev_action:
            # Experimental setting where the agent has memory of their previous action, might result in smoother steering...
            self.pre_a = 0
        else: self.pre_a = -1 #This should not occur.
    #End of __init__()

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
        
        :ownpos: is self-position (relative heading towards objective): 0-3 
            
        :opp1: is opponent 1 (relative heading towards opp1): 0-3
            
        :opp2: is opponent 2 (relative heading towards opp2): 0-3
            
        :b_flag: is own flag grabbed: Bool
            
        :r_flag: is opponent flag grabbed: Bool
            
        :action: is action space: 0-3

        :reward: is the reward that was found with the selected action

        reward is from frame n+1, the rest of the values are from frame n (action being the action between these frames, so selected in frame n).
        """
        prev = self.q_Table.prev_action # Experimental previous action memory setting.
        # old_q is taken from the (during this episode) updated q-value
        if prev:
            old_q = self.u_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][action][self.pre_a]
        else:
            old_q = self.u_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][action]

        # Calculation of loss etc as per the qlearn algorithm
        opt_future_value = -100000000000. #"." cause reward is continuous
        for i in range(4): #for every possible action do...
            # opt_future_value is taken from the un-updated qtable that is also used for action selection.
            if prev:
                opt_future_value = max(opt_future_value, self.q_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][ i ][self.pre_a])
            else:
                # (standard setting, without previous action memory)
                opt_future_value = max(opt_future_value, self.q_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][ i ])
        # Loss function is used to compute new q-value:
        new_q = (1 - self.q_Table.LEARNING_RATE) * old_q + self.q_Table.LEARNING_RATE * (reward + self.q_Table.DISCOUNT_FACTOR * opt_future_value)
        # LEARNING_RATE and other parameters should remain the same between both tables
        #print(f"Updating Q-value for state ({ownpos}, {opp1}, {opp2}, {b_flag}, {r_flag}) and action {action} from {old_q} to {new_q} based on reward {reward} and optimal future value {opt_future_value}.")
        if prev:
            self.u_Table.qtable[ownpos][opp1][opp2][b_flag][r_flag][action][self.pre_a] = new_q
            # Previous action is now set to this frames action, since set_q_value() is the last reference to it within each frame.
            self.pre_a = action
        else:
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
        #  [0.5, -135], [0.0,    0]] 
        # 90° turns are this:
        # actions = [[1.0, 0], [1.0, 90], [1.0, 180], [1.0, -90]] #(forward, right, backward, left)
        # actions = [4, 2, 0, 6] #(same actions, but as discrete indexes for pyquaticus, according to ACTION_MAP)
        # 45° turns (slightly forwards, perhaps this preserves speed?) are this:
        # [[1.0, 0], [1.0, 45], [1.0, 180], [1.0, -45]]
        # [4, 3, 0, 5]
        ''' ownpos is the relative heading towards the objective, divided into 4 areas
            pos3  |  pos0
            -----/_\-----   (agent is /_\)
            pos2  |  pos1
        '''
        # To figure out the best reward we need ownpos, opp1, opp2, b_flag, r_flag, action
        # compute ownpos:
        if obs[self.id]['has_flag']: 
            agent_heading = info[self.id]['global_state'][(self.id, "heading")]
            agent_position = info[self.id]['global_state'][(self.id, 'pos')]
            ownpos = headingToState( bearing_from_coord(FLAG_DELIVERY_POS, agent_position, agent_heading) )
        else:
            # ...and towards own base (or map half) if agent has grabbed the enemy flag
            #print("own_home_bearing =",obs[self.id]['own_home_bearing'])#output of this is "own_home_bearing = 117.06809126991149"
            ownpos = headingToState(obs[self.id]['opponent_home_bearing'])
            #print above returns 117.06809126991149 wth?!?
        if len(obs) >= 5: #3v3 detection (works)
            #not just any two opponents, we need the closest two opponents in 3v3:
            # use obs[self.id][('opponent_1', 'distance')] for comparison :-)
            if obs[self.id][('opponent_0', 'distance')] < obs[self.id][('opponent_1', 'distance')]:
                #opp0 is one of two closest opp. agents (out of three)
                opp1_bearing = headingToState(obs[self.id][('opponent_0', 'relative_heading')]) 
            else:
                #opp1 is one of two closest opp. agents
                opp1_bearing = headingToState(obs[self.id][('opponent_1', 'relative_heading')]) 
            if obs[self.id][('opponent_1', 'distance')] < obs[self.id][('opponent_2', 'distance')]:
                #opp1 is also one of two closest opp. agents
                opp2_bearing = headingToState(obs[self.id][('opponent_1', 'relative_heading')])
            else:
                #opp2 is also one of two closest opp. agents
                opp2_bearing = headingToState(obs[self.id][('opponent_2', 'relative_heading')])
            # (All closest agents should be sorted out now with the minimal number of comparisons.)
        else:
            # compute opp1 (angle (0-3) between own heading and bearing towards opponent 1)
            opp1_bearing = headingToState(obs[self.id][('opponent_0', 'relative_heading')]) 
            # compute opp2
            opp2_bearing = headingToState(obs[self.id][('opponent_1', 'relative_heading')])
        # compute b_flag (bool whether opponent has grabbed the blue flag)
        if self.q_Table.boolchange:
            b_flag = int(obs[self.id]["on_side"]) #now b_flag represents if opponents are tag-able
        else:
            if len(obs) >= 5: #TODO check if this works
            #if 3 opponents exist
                b_flag = int(obs[self.id][('opponent_0', 'has_flag')] or obs[self.id][('opponent_1', 'has_flag')] or obs[self.id][('opponent_2', 'has_flag')]) #true if any opponent has your flag Was changed to show on which side of the map we are (as parameter, but so far does not look better).
            else:
                b_flag = int(obs[self.id][('opponent_0', 'has_flag')] or obs[self.id][('opponent_1', 'has_flag')]) #true if any opponent has your flag
        # compute r_flag
        r_flag = int(obs[self.id]['has_flag']) #is boolean, but integer (0-1) is better for array index
        # loop through action (range is 0-3)
        #for loop for self.qtable[ownpos][opp1][opp2][b_flag][r_flag][i]
        q_max = -1000000000000
        a_max = -1
        for i in range(4):
            # Account for experimental setting with extra memory (and 4x increased statespace)
            if self.q_Table.prev_action:
                if q_max < self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i][self.pre_a]:
                    q_max = self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i][self.pre_a]
                    a_max = i
            # Otherwise, standard setting without previous action memory:
            else:
                if q_max < self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i]:
                    q_max = self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i]
                    a_max = i
        #actions = [[1.0, 0], [1.0, 90], [1.0, 180], [1.0, -90]] #(forward, right, backward, left) 90° turns
        if self.q_Table.sharpturns:
            actions = [4, 2, 0, 6] #(same actions, but as discrete indexes for pyquaticus, according to ACTION_MAP)
        else:
            actions = [4, 3, 0, 5] #45°turns
        return actions[a_max]
    #End of compute_action()
#End of QlearnPolicy()

def headingToState(heading: float): 
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