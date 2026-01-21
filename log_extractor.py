import re
import os
import json
import math
from collections import defaultdict

#FOLDER = "experiment_results/experiment_5rep_600sec/"
FOLDER = "experiment_results/experiment_20260119_600s_50r/"

# Regex patterns (for extracting data from the logfiles)
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
#teams (agent id) when using 3v3 agents

def dist(a, b):
    # probably there is a numpy function for this
    return math.hypot(a[0] - b[0], a[1] - b[1])

def analyze_single_log(log_address):
    # collect all positions for analysis
    blue_positions = [] # list of (x,y)
    red_positions  = []

    # Track last-known positions and tag states
    current_pos = {}
    prev_tag_state = defaultdict(lambda: False) #dictionary because more than one agent is tracked (array would be possible, performance is not a grievance here though)
    prev_pos = {}
    tag_positions = defaultdict(list) #maps agent_id to [(x, y), ...]
    bluescore = 0
    redscore = 0
    blue_def_dist = []
    red_def_dist = []
    blue_agr_dist = []
    red_agr_dist = []
    sum_distances = defaultdict(lambda: 0.)
    #for a in blue_agents + red_agents:
    #    sum_distances[a] = 0. #dont need this because defaultdict does it anyway

    with open(log_address, "r") as f:
        for line in f:

            # Position updates
            for m in pos_pattern.finditer(line):
                agent_id = int(m.group(1))
                nums = m.group(2).split(',')
                x, y = float(nums[0]), float(nums[1])

                #current_pos[agent_id] = (x, y)
                # Update distance counter and afterwards prev_pos value:
                #sum_distances[agent_id] += dist(prev_pos[agent_id], current_pos[agent_id])
                new_pos = (x, y)
                if agent_id in prev_pos:   # only accumulate after first real position
                    sum_distances[agent_id] += dist(prev_pos[agent_id], new_pos)

                prev_pos[agent_id] = new_pos
                current_pos[agent_id] = new_pos
                # collect all positions for heatmap
                if agent_id in blue_agents:
                    blue_positions.append(new_pos)
                elif agent_id in red_agents:
                    red_positions.append(new_pos)

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
            
            # m switches to flag positions
            m = flag_pattern.search(line)
            if m:
                #find current flag positions
                blue_nums = m.group(1).split(',')
                red_nums  = m.group(2).split(',')
                #print("Blue flag pos:", blue_nums) #Blue flag pos: ['20.', ' 40.'] #Red flag pos: ['140.', '  40.'] (and changing when flag is moved)
                #print("Red flag pos:", red_nums)
                blue_cords = (float(blue_nums[0]), float(blue_nums[1]))
                red_cords  = (float(red_nums[0]),  float(red_nums[1]))
                # this should only compute if all agent positions are known
                #print("\nchecking current_pos")
                if all(a in current_pos for a in blue_agents + red_agents):#i guess just to be safe
                    #print("current_pos=", current_pos)
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
                    # print("now min_r_agent is the closest red agent (id) to the blue flag.:", min_r_agent)
                    # print("min_r_dist=", min_r_dist)
                    red_agr_dist.append(min_r_dist) #(this happens to be the aggr. distance measure)

                    for blue_id in blue_agents:
                        dist_to_flag = dist(current_pos[blue_id], red_cords) #find the closest blue agent to the red flag
                        if(dist_to_flag < min_b_dist):
                            min_b_agent = blue_id
                            min_b_dist = dist_to_flag
                    #now min_b_agent is the closest blue agent (id) to the red flag.
                    # print("now min_b_agent is the closest blue agent (id) to the red flag.:", min_b_agent)
                    # print("min_b_dist=", min_b_dist)
                    blue_agr_dist.append(min_b_dist)

                    # distance from closest blue to most aggressive red 
                    min_b_dist = float("inf")
                    for blue_id in blue_agents:
                        d = dist(current_pos[blue_id], current_pos[min_r_agent])
                        if d < min_b_dist:
                            min_b_dist = d
                    # distance from closest red to most aggressive blue 
                    min_r_dist = float("inf")
                    for red_id in red_agents:
                        d = dist(current_pos[red_id], current_pos[min_b_agent])
                        if d < min_r_dist:
                            min_r_dist = d 
                    # Defensive distances aquired, append them now
                    blue_def_dist.append(min_b_dist)
                    red_def_dist.append(min_r_dist)
    # End of log file processing (open(log_address, "r") ends here)


    # Some last calculations before the results are announced
    bluetags = len(tag_positions[0]) + len(tag_positions[1]) + len(tag_positions[2])
    redtags = len(tag_positions[3]) + len(tag_positions[4]) + len(tag_positions[5])

    # Avoid division by zero
    if bluetags > 0:
        blue_score_per_tag = bluescore / bluetags
    else:
        blue_score_per_tag = 0.0

    if redtags > 0:
        red_score_per_tag = redscore / redtags
    else:
        red_score_per_tag = 0.0

    # Average Defensive Distances
    #   blue_def_dist_avg is the average distance from the closest blue agent to the most "aggressive" red agent.
    #   The most aggressive red agent is defined as the red agent closest to the blue flag.
    blue_def_dist_avg = 0
    for e in blue_def_dist:
        blue_def_dist_avg += e
    blue_def_dist_avg = blue_def_dist_avg / len(blue_def_dist) if len(blue_def_dist) > 0 else 0.0 #blue_def_dist_avg is the average distance from the closest blue agent to the most "aggressive" red agent.
    red_def_dist_avg = 0
    for e in red_def_dist:
        red_def_dist_avg += e
    red_def_dist_avg = red_def_dist_avg / len(red_def_dist) if len(red_def_dist) > 0 else 0.0 #red_def_dist_avg is the average distance from the closest red agent to the most "aggressive" blue agent.

    # Average Aggressive Distances
    #   blue_agr_dist_avg is the average distance from the closest blue agent to the red flag.
    blue_agr_dist_avg = 0
    for e in blue_agr_dist:
        blue_agr_dist_avg += e
    blue_agr_dist_avg = blue_agr_dist_avg / len(blue_agr_dist) if len(blue_agr_dist) > 0 else 0.0 #blue_agr_dist_avg is the average distance from the closest blue agent to the red flag.
    red_agr_dist_avg = 0
    for e in red_agr_dist:
        red_agr_dist_avg += e
    red_agr_dist_avg = red_agr_dist_avg / len(red_agr_dist) if len(red_agr_dist) > 0 else 0.0 #red_agr_dist_avg is the average distance from the closest red agent to the blue flag.

    # Normalizing agr and def distances by maximum possible distance (map diagonal; bigger is possible but practically not happening, especially on average)
    max_dist = math.hypot(160.0, 80.0) #map size is 160x80; might have to make this more adaptible when changing team size and other setup...
    # agr and def distances are each normalized to [0, 0.5] so their sum is normalized to [0, 1] TODO normalize more exact instead of diagonal
    blue_def_dist_norm = blue_def_dist_avg / max_dist * 0.5
    red_def_dist_norm  = red_def_dist_avg  / max_dist * 0.5
    blue_agr_dist_norm = blue_agr_dist_avg / max_dist * 0.5
    red_agr_dist_norm  = red_agr_dist_avg  / max_dist * 0.5
    blue_agrdef_dist = blue_def_dist_norm + blue_agr_dist_norm
    red_agrdef_dist  = red_def_dist_norm  + red_agr_dist_norm

    # Some more calculations but here because output is printed
    blue_total_dist = 0.0
    red_total_dist = 0.0
    for a in blue_agents + red_agents:
        #print(f"Agent {a} distance traveled: {sum_distances[a]}")
        if a in blue_agents:
            blue_total_dist += sum_distances[a]
        elif a in red_agents:
            red_total_dist += sum_distances[a]
    #print("Blue Team total distance traveled: ", blue_total_dist)   
    #print("Red Team total distance traveled: ", red_total_dist)
    
    # Prepare results dictionary
    res = dict()
    res['red_score'] = redscore
    res['blue_score'] = bluescore
    res['blue_score_per_tag'] = blue_score_per_tag
    res['red_score_per_tag'] = red_score_per_tag
    res['blue_def_avg_dist'] = blue_def_dist_avg
    res['red_def_avg_dist'] = red_def_dist_avg
    res['blue_agr_avg_dist'] = blue_agr_dist_norm
    res['red_agr_avg_dist'] = red_agr_dist_norm
    res['blue_defagr_dist'] = blue_agrdef_dist#(blue_agr_dist_norm + blue_def_dist_norm) #smaller is better, expected between [0, 1]
    res['red_defagr_dist'] = red_agrdef_dist#(red_agr_dist_norm + red_def_dist_norm)
    res['blue_total_dist'] = blue_total_dist
    res['red_total_dist'] = red_total_dist
    # save all positions (for heatmap)
    res['blue_positions'] = blue_positions
    res['red_positions'] = red_positions
    # save all tag positions
    res['tag_positions'] = dict(tag_positions)
    
    return res
# End of analyze_single_log()


def analyze_log(log_address):
    """Analyzes a log file and extracts positions, tagging positions, final scores, flag positions.
    Computes the following metrics per team,agent_type,difficulty,reward_choice combination:
    - Average distance to enemy flag (closest agent)
    - Percentage of time spent on enemy side of the map
    - Total distance traveled (all teammembers)
    - Final score
    - Average distance to the aggressive enemy agent (closest own agent to closest enemy agent to own flag)
    - Average area of the triangle formed by the three agents (team cohesion measure)
    - Positions of agents (list per team)
    - Tagging positions of agents (list per team)
    - Number of tags per team (when an agent is tagged by an enemy agent)

    The output of this function is saved in a csv for plotting and further analysis.

    Args:
        log_address (str): Path to the folder containing log files."""

    if not os.path.isdir(log_address):
        raise ValueError(f"Folder not found: {log_address}")

    # filename example: agent_type0_diffmedium_seed7935_reward1.log
    filename_re = re.compile(
        r"agent_type(?P<atype>\d+)_diff(?P<diff>[^_]+)_seed(?P<seed>\d+)_reward(?P<reward>\d+)\.log$"
    )

    metric_keys = [
        'red_score', 'blue_score',
        'blue_score_per_tag', 'red_score_per_tag',
        'blue_def_avg_dist', 'red_def_avg_dist',
        'blue_agr_avg_dist', 'red_agr_avg_dist',
        'blue_defagr_dist', 'red_defagr_dist',
        'blue_total_dist', 'red_total_dist'
    ]

    json_path = os.path.join(log_address, 'extracted_data.json')
    data = []

    for fn in sorted(os.listdir(log_address)):
        m = filename_re.match(fn)
        if not m:
            continue
        atype = m.group('atype')
        diff = m.group('diff')
        seed = m.group('seed')
        reward = m.group('reward')
        filepath = os.path.join(log_address, fn)
        try:
            res = analyze_single_log(filepath)
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue
        entry = {
            'agent_type': atype,
            'difficulty': diff,
            'seed': seed,
            'reward': reward
        }
        for key in metric_keys:
            entry[key] = res.get(key, '')
        entry['blue_positions'] = res['blue_positions']
        entry['red_positions'] = res['red_positions']
        entry['tag_positions'] = res['tag_positions']
        data.append(entry)

    with open(json_path, 'w') as jf:
        json.dump(data, jf, indent=2)

    print(f"Data extracted and saved to {json_path}")
# End of analyze_log()


if __name__ == "__main__":
    #analyze_log("experiment_results/experiment_5rep_600sec/")
    analyze_log(FOLDER)