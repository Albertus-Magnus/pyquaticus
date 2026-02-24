from pyquaticus.utils.rewards import triple_aggressive_rew, triple_caps_and_grabs
import numpy as np
import random
import ast
import re

#########################
# Q-learning own implementation to work with pyquaticus. Requires some representation of the state space and action space, as well as a reward function (or compatibility with pyquaticus reward functions). This will for now be a very basic implementation, without any opponent agents or multi-agent considerations. Just to get a working example of Q-learning in a simple environment (hopefully avoiding state space explosion). 
# Later we will extend it to opponent agents. Custom reward is also later, first use reward that is written down in training data.
#########################

# Build a q-table from a file of info and reward (etc?) logs and safe that table (is now a q-learn policy)
class QPyquaBuilder:
    def __init__(self, x=160, y=80, x_cells=8, y_cells=8):
        '''
            Input:
            x, y describe the map size. 
            x_cells is the number of cell columns in the grid divided by two (per map-half).
            y_cells is the number of cell rows in the grid.
        '''
        # As first step we must divide the map into a grid of cells.
        # The map is 160 by 80, the longer side (x-axis) is divided into the own half and the enemy half, where cells should only be located on one half at a time.
        #worldgrid = np.array([
        # Divide the map into a grid of cells, assuming x-achsis is divided into two team sides:
        self.x_cellsize = (x / 2) / x_cells # zB: 160 / 2 / 8 = 10
        self.y_cellsize = y / y_cells # zB: 80 / 8 = 10
        # Cellsize optimally should align with the border between map halves.
        cellgrid = np.zeros((x_cells, y_cells)) # This is just a grid to represent the map-dividing cells, not the actual q-table. At least for each represented agent position we need one of those for the full (but still oversimplified) state.
        # cellgrid not strictly necessary, just a reminder of the logic right now
        self.oobool = False # Necessary if position temporarily leaves the grid. Stored in addition to the map, because probably(?) only one state necessary.
        # Note: the oob-bool will also be used for the tagged-state (similar to how pyquaticus uses it anyways?)
        # TODO PROBLEM: on opponents the tagged must not be oob! (could encounter them on their map-half)
        ''' The Q-table needs account for 
                - the position of the agent (for which the rewards are tabled).
                - the positions of 2 (3?) opponents.
                - opponent flag state (holding flag or not).
            (Probably store the qtable as this-class-object for now?, not in table form)
            (Actually, we need a table because the q-values have to be stored. But we need to create that using these variables created above...)
        '''
        self.statesize = ((x_cells * 2 * y_cells + 1) ** 3) * 2
        #print("statesize=",self.statesize)
        # (Calculation assuming 2 tracked opponent agents and one oob state in addition to position.)
        # (All states exist with opponent flag grabbed (by this agent, team actions are not tracked) and with opponent flag at status quo, thus all states times two.)
        
        '''Action space consists of 8 directions at full speed and one "direction" for zero speed.'''
        self.actionsize = 9

        #q_table_size = self.statesize * self.actionsize    #This might be an enourmous size, depending on number of map grid cells :-/
        #print(self.statesize * self.actionsize)
        self.q_table = np.zeros((self.statesize, self.actionsize))
        print("Q-Table created, size",self.statesize * self.actionsize)
        #self.q_table[0][0] = 1
        #self.q_table[self.statesize-1][self.actionsize-1]
        #print("memory fine..?")
        #for i in range(self.statesize):
        #    self.q_table[i][0] = i^5 #this might be a naive test, but seems like the q-table size is fine for now...
        #print("laufzeit fine..?")

    def access_file(self, filename):
        """
        Created with Copilot to input a logfile for q-list creation/processing.
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

if __name__ == '__main__':
    q_object = QPyquaBuilder()
    logdata = q_object.access_file("experiment_results/experiment_1min_5rep_laptop/agent_type0_diffeasy_seed1603_reward1.log")
    # (Example log file, will call them all with a loop somewhere else. TODO)
    #print(logdata) #appears to be the correctly read-in file :)
    #print(list(logdata[0]['obs']['agent_0'].keys() ))
    # apparently a list of dictionaries (keys: obs,info,reward) 
    # of dictionaries (keys: agent_0, agent_1,...) of dictionaries (keys:
    #   'opponent_home_bearing', 'opponent_home_distance', 'own_home_bearing', 'own_home_distance', 'wall_0_bearing', 'wall_0_distance', 'wall_1_bearing', 'wall_1_distance', 'wall_2_bearing', 'wall_2_distance', 'wall_3_bearing', 'wall_3_distance', 'scrimmage_line_bearing', 'scrimmage_line_distance', 'speed', 'has_flag', 'on_side', 'tagging_cooldown', 'is_tagged', 'team_score', 'opponent_score', ('teammate_0', 'bearing'), ('teammate_0', 'distance'), ('teammate_0', 'relative_heading'), ('teammate_0', 'speed'), ('teammate_0', 'has_flag'), ('teammate_0', 'on_side'), ('teammate_0', 'tagging_cooldown'), ('teammate_0', 'is_tagged'), ('teammate_1', 'bearing'), ('teammate_1', 'distance'), ('teammate_1', 'relative_heading'), ('teammate_1', 'speed'), ('teammate_1', 'has_flag'), ('teammate_1', 'on_side'), ('teammate_1', 'tagging_cooldown'), ('teammate_1', 'is_tagged'), ('opponent_0', 'bearing'), ('opponent_0', 'distance'), ('opponent_0', 'relative_heading'), ('opponent_0', 'speed'), ('opponent_0', 'has_flag'), ('opponent_0', 'on_side'), ('opponent_0', 'tagging_cooldown'), ('opponent_0', 'is_tagged'), ('opponent_1', 'bearing'), ('opponent_1', 'distance'), ('opponent_1', 'relative_heading'), ('opponent_1', 'speed'), ('opponent_1', 'has_flag'), ('opponent_1', 'on_side'), ('opponent_1', 'tagging_cooldown'), ('opponent_1', 'is_tagged'), ('opponent_2', 'bearing'), ('opponent_2', 'distance'), ('opponent_2', 'relative_heading'), ('opponent_2', 'speed'), ('opponent_2', 'has_flag'), ('opponent_2', 'on_side'), ('opponent_2', 'tagging_cooldown'), ('opponent_2', 'is_tagged')
    # ) of diverse types.