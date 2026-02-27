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

#########################
# Q-learning own implementation to work with pyquaticus. Requires some representation of the state space and action space, as well as a reward function (or compatibility with pyquaticus reward functions). This will for now be a very basic implementation, without any opponent agents or multi-agent considerations. Just to get a working example of Q-learning in a simple environment (hopefully avoiding state space explosion). 
# Online-Training is the aim, the q-table can be print to file and deployed.
#########################

# Build a q-table from a file of info and reward (etc?) logs and safe that table (is now a q-learn policy)
class QPyquaBuilder:
    def __init__(self, x=160, y=80):
        '''
            Input:
            x, y describe the map size.
        '''
        #self.statesize = ((x_cells * 2 * y_cells + 1) ** 3) * 2
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

        #print(self.statesize * self.actionsize)
        self.q_table = np.zeros((self.statesize, self.actionsize))
        print("Q-Table created, size",self.statesize * self.actionsize)

    def access_file(self, filename):
        """
        Note: not the way to do training anymore. 
        Function created with Copilot to input a logfile for q-list creation/processing.
        Parse a pyquaticus .log file and return a list of frames:
        [ { 'obs': ..., 'reward': ..., 'info': ... }, ... ]
        Each block is parsed with ast.literal_eval after converting array([...]) -> [...].
        """
        frames = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f]

        def collect_block(start_idx, key):
            # start at a line containing "key:"; return (text, next_index)
            line = lines[start_idx]
            # find the position after "key:"
            pos = line.find(key + ':')
            if pos == -1:
                return None, start_idx + 1
            block = line[pos + len(key) + 1 :].lstrip()
            # count braces to handle multiline dicts
            braces = block.count('{') - block.count('}')
            i = start_idx + 1
            while braces > 0 and i < len(lines):
                part = lines[i]
                block += '\n' + part
                braces += part.count('{') - part.count('}')
                i += 1
            return block.strip(), i

        def sanitize_arrays(text):
            # replace array([...]) with [...] (handles typical numpy-style in logs)
            # non-greedy bracket capture to avoid overeating; DOTALL to include newlines
            return re.sub(r'array\(\s*(\[[^\]]*?\])\s*\)', r'\1', text, flags=re.S)

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            # look for an obs block on this line
            if 'obs:' in line:
                obs_text, i = collect_block(i, 'obs')
                # find next reward block
                while i < n and 'reward:' not in lines[i]:
                    i += 1
                if i >= n:
                    break
                reward_text, i = collect_block(i, 'reward')
                # find next info block
                while i < n and 'info:' not in lines[i]:
                    i += 1
                if i >= n:
                    break
                info_text, i = collect_block(i, 'info')

                # sanitize and try to literal_eval; if fails, keep raw strings
                try:
                    obs_parsed = ast.literal_eval(sanitize_arrays(obs_text))
                except Exception:
                    obs_parsed = obs_text
                try:
                    reward_parsed = ast.literal_eval(sanitize_arrays(reward_text))
                except Exception:
                    reward_parsed = reward_text
                try:
                    info_parsed = ast.literal_eval(sanitize_arrays(info_text))
                except Exception:
                    info_parsed = info_text

                frames.append({'obs': obs_parsed, 'reward': reward_parsed, 'info': info_parsed})
            else:
                i += 1

        return frames
    # End of access_file()

    def file_to_q_table(self, filename):
        logdata = self.access_file(filename)
        #logdata = q_object.access_file("experiment_results/experiment_5rep_600sec/agent_type0_diffeasy_seed1127_reward1.log")
        # (Example log file, will call them all with a loop somewhere else. TODO)
        #print(logdata) #appears to be the correctly read-in file :)
        #print(list(logdata[0]['obs']['agent_0'].keys() ))
        # returned ['opponent_home_bearing', 'opponent_home_distance', 'own_home_bearing', 'own_home_distance', 'wall_0_bearing', 
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
        #print(logdata[0]['obs']['agent_0']['opponent_home_bearing']) 
        # the above line returns -48.13549024484041
        # or 4.857072965975732, -173.89483662036923, 12.830624664076112, -125.44342695408903, 63.586015467142175
        # ergo from -180 to 0 to 180 is the heading range

        #print((logdata[0]['info']['agent_0']['global_state'][('agent_0', 'pos')] )) #returns [64.77070303, 23.29377587]
        # info is logdata[0]['info']['agent_0']['global_state']
        # [('agent_0', 'pos'), ('agent_0', 'heading'), ('agent_0', 'scrimmage_line_bearing'), ('agent_0', 'scrimmage_line_distance'), ('agent_0', 'speed'), ('agent_0', 'has_flag'), ('agent_0', 'on_side'), ('agent_0', 'oob'), ('agent_0', 'tagging_cooldown'), ('agent_0', 'is_tagged'), ('agent_1', 'pos'), ('agent_1', 'heading'), ('agent_1', 'scrimmage_line_bearing'), ('agent_1', 'scrimmage_line_distance'), ('agent_1', 'speed'), ('agent_1', 'has_flag'), ('agent_1', 'on_side'), ('agent_1', 'oob'), ('agent_1', 'tagging_cooldown'), ('agent_1', 'is_tagged'), ('agent_2', 'pos'), ('agent_2', 'heading'), ('agent_2', 'scrimmage_line_bearing'), ('agent_2', 'scrimmage_line_distance'), ('agent_2', 'speed'), ('agent_2', 'has_flag'), ('agent_2', 'on_side'), ('agent_2', 'oob'), ('agent_2', 'tagging_cooldown'), ('agent_2', 'is_tagged'), ('agent_3', 'pos'), ('agent_3', 'heading'), ('agent_3', 'scrimmage_line_bearing'), ('agent_3', 'scrimmage_line_distance'), ('agent_3', 'speed'), ('agent_3', 'has_flag'), ('agent_3', 'on_side'), ('agent_3', 'oob'), ('agent_3', 'tagging_cooldown'), ('agent_3', 'is_tagged'), ('agent_4', 'pos'), ('agent_4', 'heading'), ('agent_4', 'scrimmage_line_bearing'), ('agent_4', 'scrimmage_line_distance'), ('agent_4', 'speed'), ('agent_4', 'has_flag'), ('agent_4', 'on_side'), ('agent_4', 'oob'), ('agent_4', 'tagging_cooldown'), ('agent_4', 'is_tagged'), ('agent_5', 'pos'), ('agent_5', 'heading'), ('agent_5', 'scrimmage_line_bearing'), ('agent_5', 'scrimmage_line_distance'), ('agent_5', 'speed'), ('agent_5', 'has_flag'), ('agent_5', 'on_side'), ('agent_5', 'oob'), ('agent_5', 'tagging_cooldown'), ('agent_5', 'is_tagged'), 'blue_flag_home', 'red_flag_home', 'blue_flag_pos', 'red_flag_pos', 'blue_flag_pickup', 'red_flag_pickup', 'blue_team_score', 'red_team_score']
        # apparently a list of dictionaries (keys: obs,info,reward) 
        # of dictionaries (for obs!) (keys: agent_0, agent_1,...) of dictionaries (keys:
        #   'opponent_home_bearing', 'opponent_home_distance', 'own_home_bearing', 'own_home_distance', 'wall_0_bearing', 'wall_0_distance', 'wall_1_bearing', 'wall_1_distance', 'wall_2_bearing', 'wall_2_distance', 'wall_3_bearing', 'wall_3_distance', 'scrimmage_line_bearing', 'scrimmage_line_distance', 'speed', 'has_flag', 'on_side', 'tagging_cooldown', 'is_tagged', 'team_score', 'opponent_score', ('teammate_0', 'bearing'), ('teammate_0', 'distance'), ('teammate_0', 'relative_heading'), ('teammate_0', 'speed'), ('teammate_0', 'has_flag'), ('teammate_0', 'on_side'), ('teammate_0', 'tagging_cooldown'), ('teammate_0', 'is_tagged'), ('teammate_1', 'bearing'), ('teammate_1', 'distance'), ('teammate_1', 'relative_heading'), ('teammate_1', 'speed'), ('teammate_1', 'has_flag'), ('teammate_1', 'on_side'), ('teammate_1', 'tagging_cooldown'), ('teammate_1', 'is_tagged'), ('opponent_0', 'bearing'), ('opponent_0', 'distance'), ('opponent_0', 'relative_heading'), ('opponent_0', 'speed'), ('opponent_0', 'has_flag'), ('opponent_0', 'on_side'), ('opponent_0', 'tagging_cooldown'), ('opponent_0', 'is_tagged'), ('opponent_1', 'bearing'), ('opponent_1', 'distance'), ('opponent_1', 'relative_heading'), ('opponent_1', 'speed'), ('opponent_1', 'has_flag'), ('opponent_1', 'on_side'), ('opponent_1', 'tagging_cooldown'), ('opponent_1', 'is_tagged'), ('opponent_2', 'bearing'), ('opponent_2', 'distance'), ('opponent_2', 'relative_heading'), ('opponent_2', 'speed'), ('opponent_2', 'has_flag'), ('opponent_2', 'on_side'), ('opponent_2', 'tagging_cooldown'), ('opponent_2', 'is_tagged')
        # ) of diverse types.
        #DONE read through frames and edit table
        #DONE create 2v2 training data or adapt to 3v3 [we will use 3v3 training data but ignore the third opponent for now, got a ArrayMemoryError with 3 opponents...]
        #then train, then eod :)

        #right now hardcoded to read for agent_0 only (and 2 opponents, ignoring 1 opponent to utilize existing logs - 3 opponents was too large)
        previous_reward = logdata[0]['reward']['agent_0'] #set to first frame reward (should have same effect as skipping first frame)
        prev_xcell = int(logdata[0]['info']['agent_0']['global_state'][('agent_0', 'pos')][0] / self.x_cellsize)
        prev_ycell = int(logdata[0]['info']['agent_0']['global_state'][('agent_0', 'pos')][0] / self.x_cellsize) #first position shouldn't be oob...
        for frame in logdata:
            obs = frame['obs']['agent_0']
            reward = frame['reward']['agent_0'] 
            info = frame['info']['agent_0']['global_state']
            #print(obs)
            #print(reward)
            #print(info)
            # Needed to decide qtable id: ('agent_0', 'has_flag')('agent_0', 'oob')('agent_0', 'is_tagged') (and below ones)
            # edit q-table with this data (need to create a state representation first, then find the action taken, then update the q-table with the reward and learning rate, etc.)
            has_flag = info[('agent_0', 'has_flag')]
            self_oob = info[('agent_0', 'oob')]
            is_tagged = info[('agent_0', 'is_tagged')]
            heading = info[('agent_0', 'heading')]
            speed = info[('agent_0', 'speed')]

            #sort location to grid:
            x_cell = int( info[('agent_0', 'pos')][0] / self.x_cellsize )
            # x / cellwidth (round down) is the index of the cell from "left to right" (x-achsis cell id)
            # after the last cell one index is reserved for oob and (if not opponent) tag state. (if x,y coded is x_max+1,y_max)
            y_cell = int( info[('agent_0', 'pos')][1] / self.y_cellsize )
            if info[('agent_0', 'oob')] or is_tagged == True:
                x_cell = self.x_cells # oob state is one x cell after the last cell, y_cell is last y cell + 0
                y_cell = self.y_cells - 1 #oob (or tagged, is tracked with same state since loss of control)

            #opponent data (just position+oob)
            x_cell_opp1 = int( info[('agent_3', 'pos')][0] / self.x_cellsize )
            y_cell_opp1 = int( info[('agent_3', 'pos')][1] / self.y_cellsize )
            if info[('agent_3', 'oob')] == True:
                x_cell_opp1 = self.x_cells # oob state is one x cell after the last cell, y_cell is last y cell + 0
                y_cell_opp1 = self.y_cells - 1 #oob
            x_cell_opp2 = int( info[('agent_4', 'pos')][0] / self.x_cellsize )
            y_cell_opp2 = int( info[('agent_4', 'pos')][1] / self.y_cellsize )
            if info[('agent_4', 'oob')] == True:
                x_cell_opp2 = self.x_cells # oob state is one x cell after the last cell, y_cell is last y cell + 0
                y_cell_opp2 = self.y_cells - 1 #oob
            
            #TODO heading and speed in the info-log are mapped in some strange way to the ACTION_MAP I took from elsewhere in the code.     [best way: print action and print info with step number. Then look]
            #Figure out how to accurately find the according action. For now this is a bit sus implemented (to test if the rest works).
            # Heading and speed (for now, not optimal solution!) are mapped to the closest action in the ACTION_MAP:
            ACTION_MAP = {
                0: [0.0,    0],
                1: [1.0,    0],
                2: [1.0,   45],
                3: [1.0,   90],
                4: [1.0,  135],
                5: [1.0,  180],
                6: [1.0, -135],
                7: [1.0,  -90],
                8: [1.0,  -45]
            }
            action_taken = None
            for action_id, (a_speed, a_heading) in ACTION_MAP.items():
                if a_speed > 0.5 and abs(a_heading - heading) < (45 / 2): #some tolerance for mapping (currently), not optimal but should work for now
                    action_taken = action_id
                    break
                if a_speed < 0.5:
                    action_taken = 0
            # need to find out how action heading works, also speed (is max speed 10.0 or 1.0 on input and what on output/log?)
            if action_taken is None:
                print("Warning: action not found for heading", heading, "and speed", speed)
                action_taken = 0 # default to no movement if not found (should not happen)

            # Compute state id for q-table: (concerns the state of the previous frame)
            #location_space_size = q_object.x_cells * q_object.y_cells + 1 # +1 for oob state
            state_id = prev_xcell + prev_ycell * self.x_cells + \
            x_cell_opp1 + y_cell_opp1 * self.x_cells * (self.x_cells + 1) + \
            x_cell_opp2 + y_cell_opp2 * self.x_cells * (self.x_cells + 1) * (self.x_cells + 1) + \
            int(has_flag) * (self.x_cells + 1) ** 3
            # Does this encoding make sense? Idk, my head is smoking now. This will be a nightmare if I have to look at it again, perhaps a dictionary would be better (but not for memory & performance?)

            # Using reward['agent_0'] and previous_reward, update q-table with learning rate alpha and discount factor gamma:
            alpha = 0.1 # learning rate
            gamma = 0.9 # discount factor
            reward_value = reward - previous_reward # this is the reward for the current action
            old_q_value = self.q_table[state_id][action_taken]
            # Q-learning update rule:
            self.q_table[state_id][action_taken] = old_q_value + alpha * (reward_value + gamma * np.max(self.q_table[state_id]) - old_q_value)
            #TODO check correctness of update rule (lookup B. loss function)

            #set previous-values for next frame:
            previous_reward = reward
            prev_xcell = x_cell
            prev_ycell = y_cell
        #return #No return necessary, q_object is already the one used in main and is updated.
    #End of file_to_q_table()

    def lookup_action(self, obs, info: dict[str, dict]):
        #what do i need to compute q-value address?
        '''has_flag = info[('agent_0', 'has_flag')]
            self_oob = info[('agent_0', 'oob')]
            is_tagged = info[('agent_0', 'is_tagged')]
            heading = info[('agent_0', 'heading')]
            speed = info[('agent_0', 'speed')]
            positions of all 3 agents, including oob option
            That should be all.'''
        return 16 #TODO write this function 
               # (extract all necessary variables and compute the correct cell(s) of qtable to lookup qvalues, find maximum action-qvalue cell and return that action)
#End of QPyquaBuilder()

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
    #x_length = 160
    #y_length = 80
    q_object = QPyquaBuilder()
    # train qtable on one single file (could do this multiple times in a row, with multiple files but here just testing)
    q_object.file_to_q_table("experiment_results/experiment_1min_5rep_laptop/agent_type0_diffeasy_seed1603_reward1.log")

    # Q-table can now be saved and used for action selection in a policy.
    #storing q-table as a numpy file for now, can be loaded in a policy class later:
    np.save("q_table.npy", q_object.q_table)
    print(q_object.q_table) #does not print whole table
    
    import matplotlib.pyplot as plt

    # Visualize the q-table
    #plt.figure(figsize=(12, 6))
    #plt.imshow(q_object.q_table, cmap='viridis', aspect='auto')
    #plt.colorbar(label='Q-value')
    #plt.xlabel('Action')
    #plt.ylabel('State')
    #plt.title('Q-Table Visualization')
    #plt.tight_layout()
    #plt.savefig('q_table_visualization.png', dpi=150)
    #plt.show() #also is not very helpful, need to find a better way to visualize it, if that is necessary... (TODO?) (just make a graph where non-zero entries are big dots)
