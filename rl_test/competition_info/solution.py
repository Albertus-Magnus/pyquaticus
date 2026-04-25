import numpy as np
import os
from ray.rllib.policy.policy import Policy
#Need an added import for competition submission?
#Post an issue to the github and we will work to get it added into the system!

#NOTE: You are only allowed to change the gen_config OBS params specified
# Changing additional variables will result in disqualification of that entry

#YOUR CODE HERE

#Load in your trained model and return the corresponding agent action based on the information provided in step()
class solution:
	#Add Variables required for solution
	
    def __init__(self):
        
        #Load in policy or anything else you want to load/do here
        #NOTE: You can only load from files that are in the same directory as the solution.py or a subdirectory
        
        #Load in learned policies see examples below:
		#Example Path: Policy.from_checkpoint(os.path.abspath('./working_dir/'+'iter_0/policies/agent-0-policy/'))
        # self.policy_one = Policy.from_checkpoint(os.path.abspath('./working_dir/' + '<Your Policy Path Here>'))
        # self.policy_two = Policy.from_checkpoint(os.path.abspath('./working_dir/' + '<Your Policy Path Here>'))
        # self.policy_three = Policy.from_checkpoint(os.path.abspath('./working_dir/' + '<Your Policy Path Here>'))
        #blue_qtable_array = np.load((os.path.abspath('./working_dir/' + 'enhancegrablong3_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_5nrs_300ep_no_pre_nr0_q_table_i174.npy')))
        blue_qtable_array = np.load((os.path.abspath('./working_dir/' + 'qtable.npy'))) 
        # blue_qtable_array = np.load((os.path.abspath('./rl_test/competition_info/' + 'qtable.npy'))) 
        # print("Q-Table loaded, shape:", np.shape(blue_qtable_array))
        # we actually only need one qtable, since only the locations in compute_action() have to be different...

        ##########
        boolchange = True #set for current qtable, adjust when using different qtable (was trained specifically for one of either choice)
        self.sharpturns = False #set for 25-04-26 qtable
        ##########
        LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE = 0.1, 0.9, 10.0 #we probably can remove those from here
        self.q_Table = QTable(LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, blue_qtable_array, boolchange) 
        # self.red_qtable = QTable()
        # self.qpolicy = QlearnPolicy("agent_0", None, self.blue_and_red_qtable, self.blue_and_red_qtable)

	
    
    
    #Given an observation return a valid action agent_id is agent that needs an action, observation space is the current normalized observation space for the specific agent
    def compute_action(self, agent_id:str, full_obs_normalized:dict, full_obs:dict, global_state:dict):
        #WARNING: If using global state you must ensure your entry can run on both RED and BLUE sides
        # State includes actual coordinate positions which are not the same on each side
        # return 0 #Remove this Line and replace the lines below with your implementation
        # if agent_id == 'agent_0' or agent_id == 'agent_3': 
        #     return self.policy_one.compute_single_action(full_obs_normalized[agent_id], explore=False)[0]
        # elif agent_id == 'agent_1' or agent_id == 'agent_4':
        #     return self.policy_two.compute_single_action(full_obs_normalized[agent_id], explore=False)[0]
        # else:
        #     return self.policy_three.compute_single_action(full_obs_normalized[agent_id], explore=False)[0]
        
        # Returns an action index (since we are aiming at discrete agent implementation), according to the ACTION_MAP from config:
        # ACTION MAP:
        # [[1.0,  180], [1.0,  135], [1.0,  90], 
        #  [1.0,   45], [1.0,    0], [1.0, -45], 
        #  [1.0,  -90], [1.0, -135], [0.5, 180], 
        #  [0.5,  135], [0.5,   90], [0.5,  45], 
        #  [0.5,    0], [0.5,  -45], [0.5, -90], 
        #  [0.5, -135], [0.0,    0]] 
        ''' ownpos is the relative heading towards the objective, divided into 4 areas
            pos3  |  pos0
            -----/_\-----   (agent is /_\)
            pos2  |  pos1
        '''
        # To figure out the best reward we need ownpos, opp1, opp2, b_flag, r_flag
        # compute ownpos:
        if full_obs[agent_id]['has_flag']:

            # agent_heading = global_state[agent_id]['global_state'][(agent_id, 'heading')] #deactivated to test if other address works, since this might be the issue encountered in step 97...
            agent_heading = global_state[(agent_id, 'heading')] #this is how global_state should be accessed.
            # agent_position = global_state[agent_id]['global_state'][(agent_id, 'pos')] #deactivated to test if other address works, since this might be the issue encountered in step 97...
            agent_position = global_state[(agent_id, 'pos')]

            if agent_id in ['agent_0', 'agent_1', 'agent_2']:
                FLAG_DELIVERY_POS = B_FLAG_DELIVERY_POS
            else:
                FLAG_DELIVERY_POS = R_FLAG_DELIVERY_POS
            ownpos = headingToState( bearing_from_coord(FLAG_DELIVERY_POS, agent_position, agent_heading) )
        else:
            ownpos = headingToState(full_obs[agent_id]['opponent_home_bearing'])
        #not just any two opponents, we need the closest two opponents in 3v3:
        # use obs[self.id][('opponent_1', 'distance')] for comparison :-)
        #full_obs[agent_id]
        #if obs[self.id][('opponent_0', 'distance')] < obs[self.id][('opponent_1', 'distance')]:
        if full_obs[agent_id][('opponent_0', 'distance')] < full_obs[agent_id][('opponent_1', 'distance')]:
            #opp0 is one of two closest opp. agents (out of three)
            opp1_bearing = headingToState(full_obs[agent_id][('opponent_0', 'relative_heading')]) 
        else:
            #opp1 is one of two closest opp. agents
            opp1_bearing = headingToState(full_obs[agent_id][('opponent_1', 'relative_heading')]) 
        if full_obs[agent_id][('opponent_1', 'distance')] < full_obs[agent_id][('opponent_2', 'distance')]:
            #opp1 is also one of two closest opp. agents
            opp2_bearing = headingToState(full_obs[agent_id][('opponent_1', 'relative_heading')])
        else:
            #opp2 is also one of two closest opp. agents
            opp2_bearing = headingToState(full_obs[agent_id][('opponent_2', 'relative_heading')])
        # (All closest agents should be sorted out now with the minimal number of comparisons.)
        # compute b_flag (bool whether opponent has grabbed the blue flag)
        if self.q_Table.boolchange:
            b_flag = int(full_obs[agent_id]["on_side"]) #now b_flag represents if opponents are tag-able
        else:
            #if 3 opponents exist
            b_flag = int(full_obs[agent_id][('opponent_0', 'has_flag')] or full_obs[agent_id][('opponent_1', 'has_flag')] or full_obs[agent_id][('opponent_2', 'has_flag')]) #true if any opponent has your flag Was changed to show on which side of the map we are (as parameter, but so far does not look better).
            # else:
            #     b_flag = int(full_obs[agent_id][('opponent_0', 'has_flag')] or full_obs[agent_id][('opponent_1', 'has_flag')]) #true if any opponent has your flag
        # compute r_flag
        r_flag = int(full_obs[agent_id]['has_flag']) #is boolean, but integer (0-1) is better for array index
        # loop through action (range is 0-3)
        #for loop for self.qtable[ownpos][opp1][opp2][b_flag][r_flag][i]
        q_max = -1000000000000
        a_max = -1
        for i in range(4):
            if q_max < self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i]:
                q_max = self.q_Table.qtable[ownpos][opp1_bearing][opp2_bearing][b_flag][r_flag][i]
                a_max = i
        #print("Maximum reward",q_max,"expected for action",i,".")
        #return a_max #translate first to pyquaticus action
        #actions = [[1.0, 0], [1.0, 90], [1.0, 180], [1.0, -90]] #(forward, right, backward, left)
        if self.sharpturns:
            actions = [4, 2, 0, 6] #(same actions, but as discrete indexes for pyquaticus, according to ACTION_MAP)
        else:
            actions = [4, 3, 0, 5]
        return actions[a_max]
    #End of compute_action()




#End of solution class

B_FLAG_DELIVERY_POS = np.array([10., 70.]) # (upper left corner)
R_FLAG_DELIVERY_POS = np.array([150., 70.]) # (upper right corner)

class QTable:
    def __init__(self, LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_Q_VALUE, qtablearray: np.ndarray, boolchange):
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.INITIAL_Q_VALUE = INITIAL_Q_VALUE
        
        # the only two important steps:
        self.boolchange = boolchange
        self.qtable = qtablearray
    #End of init()

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

def headingDiff(heading, pos1, pos2):
        # heading can range from -180 to 180, we want to compute with 360-range values
        heading = heading + 180.0
        #heading += 0.0000001 #add this if float leads to very small negative values...
        #compute angle (relative to coordinate system north) from pos1 to pos2
        angle = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) * 180 / np.pi
        #compute difference between heading and angle
        diff = (heading - angle) % 360
        diff = diff - 180.0 
        return diff


def angle180(deg):
    """Rotates an angle to be between -180 and +180 degrees."""
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg

def heading_angle_conversion(deg):
    """
    Converts a world-frame angle to a heading and vice-versa
    The transformation is its own inverse
    Args:
        deg: the angle (heading) in degrees
    Returns:
        float: the heading (angle) in degrees.
    """
    return (90 - deg) % 360

def degrees_np(rad):
    return rad * (180.0 / np.pi)

def atan2_np(y, x):
    return np.arctan2(y, x)

def vec_to_heading(vec):
    """Converts a vector to a magnitude and heading (deg)."""
    # import math #not sure if I'm allowed to import math here, using np only instead...
    angle = degrees_np(atan2_np(vec[1], vec[0]))
    # math.degrees(math.atan2(vec[1], vec[0]))
    return angle180(heading_angle_conversion(angle))

def bearing_from_coord(goal_pos, agent_pos, agent_heading):
    """Computes the relative bearing for a given agent towards a global position/coordinate.
    agent_heading is the current direction (global) the agent is directed/headed/turned 
    towards, in degrees between -180 and 180.
    for example: info['agent_0']['global_state'][(agentID, "heading")] """
    vec_from_agent = np.array([goal_pos[0] - agent_pos[0] , goal_pos[1] - agent_pos[1]])
    return angle180(vec_to_heading(vec_from_agent) - agent_heading)


#END OF CODE SECTION
