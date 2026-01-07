import re
#import matplotlib.pyplot as plt
from collections import defaultdict
import math

# Regex patterns
pos_pattern = re.compile(
    r"\('agent_(\d+)', 'pos'\)\s*:\s*array\(\s*\[([^\]]+)\]\s*\)"
)
tag_pattern = re.compile(
    r"\('agent_(\d+)', 'is_tagged'\)\s*:\s*(True|False)"
)
score_pattern = re.compile(
    r"'blue_team_score':\s*(\d+),\s*'red_team_score':\s*(\d+)" #matches 'blue_team_score': 0, 'red_team_score': 0
)
flag_pattern = re.compile(
    r"'blue_flag_pos'\s*:\s*array\(\s*\[([^\]]+)\]\s*\)\s*,\s*'red_flag_pos'\s*:\s*array\(\s*\[([^\]]+)\]\s*\)" #matches 'blue_flag_pos': array([20., 40.]), 'red_flag_pos': array([140., 40.])
)

blue_agents = [0, 1, 2]
red_agents  = [3, 4, 5]

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Track last-known positions and tag states
current_pos = {}                    # agent_id -> (x, y)
prev_tag_state = defaultdict(lambda: False) #dictionary because more than one agent is tracked
prev_pos = defaultdict(lambda: [0.,0.])
tag_positions = defaultdict(list)   # agent_id -> [(x, y), ...]
bluescore = 0
redscore = 0
blue_def_dist = []
red_def_dist = []
sum_distances = defaultdict(lambda: 0.)
for a in blue_agents + red_agents:
    sum_distances[a] = 0. #is a dictionary efficient enough here? ...probably

with open("match.log", "r") as f:
    for line in f:

        # Position updates
        for m in pos_pattern.finditer(line):
            agent_id = int(m.group(1))
            nums = m.group(2).split(',')
            x, y = float(nums[0]), float(nums[1])
            current_pos[agent_id] = (x, y)
            # Update distance counter and afterwards prev_pos value:
            sum_distances[agent_id] += dist(prev_pos[agent_id], current_pos[agent_id])

        # Tag updates
        for m in tag_pattern.finditer(line):
            agent_id = int(m.group(1))
            is_tagged = (m.group(2) == "True")

            # Detect False to True transition
            if (not prev_tag_state[agent_id]) and is_tagged and agent_id in current_pos:
                    tag_positions[agent_id].append(current_pos[agent_id])

            prev_tag_state[agent_id] = is_tagged

        # grab score (overwrite every frame, so score of last frame will be final value)
        for m in score_pattern.finditer(line):
             bluescore = int(m.group(1)) #first capture group contains blue team score
             redscore = int(m.group(2)) #second capture group contains blue team score
        
        m = flag_pattern.search(line)
        if m:
            #find current flag positions
            blue_cords = m.group(1).split(',')
            red_cords  = m.group(2).split(',')
            # this should only compute if all agent positions are known
            if all(a in current_pos for a in blue_agents + red_agents):#i guess just to be safe
                min_r_agent = -1
                min_r_dist = 2000000.0
                closest_to_r = 0
                min_b_agent = -1
                min_b_dist = 2000000.0
                closest_to_b = 3

                for red_id in red_agents:
                    dist_to_flag = dist(current_pos[red_id], blue_cords) #find the closest red agent to the blue flag
                    if(dist_to_flag < min_r_dist):
                        min_r_agent = red_id
                        min_r_dist = dist_to_flag
                #now min_r_agent is the closest red agent (id) to the blue flag.

                for blue_id in blue_agents:
                    dist_to_flag = dist(current_pos[blue_id], red_cords) #find the closest blue agent to the red flag
                    if(dist_to_flag < min_b_dist):
                        min_b_agent = blue_id
                        min_b_dist = dist_to_flag

                min_b_dist = dist(current_pos[closest_to_r], current_pos[min_r_agent]) #set to first distance to be compared (computes twice but should be efficient enough...)
                for blue_id in blue_agents:
                    dist_to_agent = dist(current_pos[blue_id], current_pos[min_r_agent]) #find the closest blue agent to the "aggressive" or "dangerous" red agent
                    if(dist_to_agent < min_b_dist):
                        min_b_dist = dist_to_agent
                #now min_b_dist is the distance from the closest blue agent to the most "aggressive" red agent.

                min_r_dist = dist(current_pos[closest_to_r], current_pos[min_r_agent]) #set to first distance to be compared 
                for blue_id in blue_agents:
                    dist_to_agent = dist(current_pos[blue_id], current_pos[min_r_agent]) #find the closest red agent to the "aggressive" or "dangerous" blue agent
                    if(dist_to_agent < min_r_dist):
                        min_r_dist = dist_to_agent
                #now min_r_dist is the distance from the closest red agent to the most "aggressive" blue agent.
                
                blue_def_dist.append(min_b_dist)
                red_def_dist.append(min_r_dist)

        

# Some last calculations
bluetags = len(tag_positions[0]) + len(tag_positions[1]) + len(tag_positions[2])
redtags = len(tag_positions[3]) + len(tag_positions[4]) + len(tag_positions[5])
blue_def_dist_avg = 0
for e in blue_def_dist:
    blue_def_dist_avg += e
blue_def_dist_avg = blue_def_dist_avg / len(blue_def_dist) #blue_def_dist_avg is the average distance from the closest blue agent to the most "aggressive" red agent.
red_def_dist_avg = 0
for e in red_def_dist:
    red_def_dist_avg += e
red_def_dist_avg = red_def_dist_avg / len(red_def_dist)

# Outputting results
print("--- Results: ---")
#The results need to be saved at some point in development/testing. I will figure out the best way once I need it...
print("BLUE Team score/tags (indicates defensive capability of RED agents): ",bluescore/bluetags) #maybe rename, also might be interesting only tags of flagbearer agents
print("Red Team score/tags (indicates defensive capability of blue agents): ",redscore/redtags)
#print(f"Red agent {red_id}: closest blue is {def_blue_agent} at distance {def_dist:.3f}")
print("Blue Team avg. defensive distance: ",blue_def_dist_avg)
print("Red Team avg. defensive distance: ",red_def_dist_avg)

for a in blue_agents + red_agents:
    print(f"Agent {a} distance traveled: {sum_distances[a]}") #do I want to normalize this by time/steps? or perhaps as a team-wide number? if so, average per team of average dist per agent? or sum / 3?
