from datetime import datetime
import re
import os
import json
import math
from collections import defaultdict
import sys
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

FOLDER = "experiment_results/experiment_5rep_600sec/"
#FOLDER = "experiment_results/experiment_20260119_600s_50r/"

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
side_pattern = re.compile(
    r"\('agent_(\d+)',\s*'on_side'\):\s*(True|False)," #matches ('agent_0', 'on_side'): True,
)

blue_agents = [0, 1, 2]
red_agents  = [3, 4, 5]
#teams (agent id) when using 3v3 agents

def dist(a, b):
    # probably there is a numpy function for this
    return math.hypot(a[0] - b[0], a[1] - b[1])

def triangle_area(p1, p2, p3):
    # a = dist(p1, p2)
    # b = dist(p2, p3)
    # c = dist(p3, p1)
    # s = 0.5 * (a + b + c)
    # return math.sqrt(s * (s - a) * (s - b) * (s - c))
    return (abs((p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1]) - (p2[0]*p1[1] + p3[0]*p2[1] + p1[0]*p3[1]))) / 2
    # hopefully this is a correct triangle formula now, should be shoelace formula...

def triangle_area2(a, b, c):
    # Since we have to compute the sides of the triangle anyways for the mean 
    # dist coverage measure (at least until one of the coverage measures is 
    # retired), we can use the old formula.
    s = 0.5 * (a + b + c)
    return math.sqrt(s * (s - a) * (s - b) * (s - c))
    #return (abs((p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1]) - (p2[0]*p1[1] + p3[0]*p2[1] + p1[0]*p3[1]))) / 2
    # hopefully this is a correct triangle formula now, should be shoelace formula...

def voronoi_uniformity(points):
    # Arena bounds
    arena = Polygon([(0,0), (160,0), (160,80), (0,80)])
    vor = Voronoi(points)

    # Voronoi-Helper
    def voronoi_cells(vor, radius=1e6):
        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        all_ridges = {}

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]

            if all(v >= 0 for v in region):
                new_regions.append(region)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = [v for _, v in sorted(zip(angles, new_region))]

            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)

    regions, vertices = voronoi_cells(vor)

    areas = []

    for region in regions:
        poly = Polygon(vertices[region])
        poly = poly.intersection(arena)
        if not poly.is_empty:
            areas.append(poly.area)

    areas = np.array(areas)

    if len(areas) == 0:
        print("voronoi error areas are zero")
        return np.nan

    uniformity = areas.std() / areas.mean()

    #print("Cell areas:", areas)
    #print("Uniformity (CV):", uniformity)
    return uniformity
# End of voronoi_uniformity()

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
    blue_triangle_areas = []
    red_triangle_areas = []
    blue_cover_dist = []
    red_cover_dist = []
    # red/blue_sides stores a list of the side-booleans for the teams.
    # True means an agent is on his teams side, every frame for each 
    # agent a value is stored (to later compute the percentage of 
    # positions that is on-side/off-side)
    blue_sides = []
    red_sides = []
    #for a in blue_agents + red_agents:
    #    sum_distances[a] = 0. #dont need this because defaultdict does it anyway
    blue_voronoi = []
    red_voronoi = []

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
                    # Compute which side the agent is on and store the value
                    if x <= 80.:
                        blue_sides.append(True)
                    else:
                        blue_sides.append(False)
                elif agent_id in red_agents:
                    red_positions.append(new_pos)
                    # Compute which side the agent is on and store the value
                    if x <= 80.:
                        red_sides.append(False)
                    else:
                        red_sides.append(True)
                # Compute which side the agent is on and store the value
                #True -> agent is on his teams half of the arena"""

            # Compute area of triangle formed by each team (for cohesion measure)
            # print(len(current_pos)) #6 elements in current_pos, is overwritten every frame of the analysis
            # TODO improve performance DONE but still bad :/
            # In addition, there is the mean distance between agents that also has to be computed for the spatial coverage score (mean dist variant)
            if all(a in current_pos for a in blue_agents):
                p0 = current_pos[blue_agents[0]]
                p1 = current_pos[blue_agents[1]]
                p2 = current_pos[blue_agents[2]]
                a = dist(p0, p1)
                b = dist(p1, p2)
                c = dist(p2, p0)
                #area = triangle_area(p0, p1, p2)
                area = triangle_area2(a, b, c)
                blue_triangle_areas.append(area)
                #blue_cover_dist.append(a + b + c)
                # Normalize mean-dist-cover by 80+80+113.14 (longest imaginable mean distance between team members in 3v3
                #blue_cover_dist.append(1 - (273.14 / (a + b + c)))
                blue_cover_dist.append(((a + b + c) / 273.14))
                # Voronoi uniformity:
                points = np.array([p0,p1,p2])
                blue_voronoi.append(voronoi_uniformity(points))
            if all(a in current_pos for a in red_agents):
                p0 = current_pos[red_agents[0]]
                p1 = current_pos[red_agents[1]]
                p2 = current_pos[red_agents[2]]
                a = dist(p0, p1)
                b = dist(p1, p2)
                c = dist(p2, p0)
                #area = triangle_area(p0, p1, p2)
                area = triangle_area2(a, b, c)
                red_triangle_areas.append(area)
                red_cover_dist.append((a + b + c)/ 273.14)
                # Voronoi uniformity:
                points = np.array([p0,p1,p2])
                red_voronoi.append(voronoi_uniformity(points))

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

            """for m in side_pattern.finditer(line):
                if not bool(m.group(2)):
                    #print("BESTÄTIGE EIN AGENT NICHT AUF WAHRER SEITE!") <-on_side is just a selector where the agents spawn, have to compute the side manually here.
                #note down which side which agent is on to later compute percentage 
                #print(m.group(1),"; ",m.group(2),"; ",blue_agents) #->0 is the entire pattern, 1 and 2 are the single values
                if int(m.group(1)) in blue_agents:
                    blue_sides.append(bool(m.group(2)))
                elif int(m.group(1)) in red_agents:
                    red_sides.append(bool(m.group(2)))
                    #True -> agent is on his teams halth of the arena"""
            
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
    blue_agrdef_dist = 1. - (blue_def_dist_norm + blue_agr_dist_norm)
    red_agrdef_dist  = 1. - (red_def_dist_norm  + red_agr_dist_norm)
    # Currently being tested is 1 - blue_agrdef_dist instead of normal, so it is better when bigger and mmore direct comparison to other measures possible

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
    # def and agr distances are all stored as normed versions
    res['blue_def_avg_dist'] = blue_def_dist_norm
    res['red_def_avg_dist'] = red_def_dist_norm
    res['blue_agr_avg_dist'] = blue_agr_dist_norm
    res['red_agr_avg_dist'] = red_agr_dist_norm
    res['blue_defagr_dist'] = blue_agrdef_dist#(blue_agr_dist_norm + blue_def_dist_norm) #smaller is better, expected between [0, 1]
    res['red_defagr_dist'] = red_agrdef_dist#(red_agr_dist_norm + red_def_dist_norm)
    res['blue_total_dist'] = blue_total_dist
    res['red_total_dist'] = red_total_dist
    blue_triangle_area_avg = sum(blue_triangle_areas) / len(blue_triangle_areas) if blue_triangle_areas and len(blue_triangle_areas) > 0 else 0.0
    # blue_triangle_area_avg is normalized by arena-area * 0.5 because that is the largest triangle-coverage expected from the agents
    blue_triangle_area_avg = blue_triangle_area_avg / (80.0 * 160.0 * 0.5)
    red_triangle_area_avg = sum(red_triangle_areas) / len(red_triangle_areas) if red_triangle_areas and len(red_triangle_areas) > 0 else 0.0    
    # blue_triangle_area_avg is normalized by arena-area * 0.5 because that is the largest triangle-coverage expected from the agents
    red_triangle_area_avg = red_triangle_area_avg / (80.0 * 160.0 * 0.5) 
    res['blue_triangle_area'] = blue_triangle_area_avg
    res['red_triangle_area'] = red_triangle_area_avg
    # Another approach at a team-wide spatial coverage score: average added distance between the teammembers
    blue_cover_dist_avg = sum(blue_cover_dist) / len(blue_cover_dist) if blue_cover_dist and len(blue_cover_dist) > 0 else 0.0
    red_cover_dist_avg = sum(red_cover_dist) / len(red_cover_dist) if red_cover_dist and len(red_cover_dist) > 0 else 0.0
    res['blue_cover_dist'] = blue_cover_dist_avg
    res['red_cover_dist'] = red_cover_dist_avg
    # save all positions (for heatmap)
    res['blue_positions'] = blue_positions
    res['red_positions'] = red_positions
    # save all tag positions
    res['tag_positions'] = dict(tag_positions)
    # save percentage of team positions on own side of the arena
    #print("blue_sides: ",blue_sides)
    #print("red_sides: ",red_sides)
    count = 0
    for side in blue_sides:
        if side: #side has to be false for agent to be on own side? does it work different than expected? <-No, the error was elsewhere
            count += 1
    blue_perc = count / float(len(blue_sides))
    count = 0
    for side in red_sides:
        if side:
            count += 1
    red_perc = count / float(len(red_sides))
    res['blue_sides'] = blue_perc
    res['red_sides'] = red_perc
    # Greater own-side-percentage means team is playing more defensive, lesser means more aggressive.
    # (percentage here is still z.B. 0.8 for 80%)
    res['blue_voronoi'] = blue_voronoi
    res['red_voronoi'] = red_voronoi
    
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
        'blue_total_dist', 'red_total_dist',
        'blue_triangle_area', 'red_triangle_area',
        'blue_cover_dist', 'red_cover_dist',
        'blue_sides', 'red_sides',
        'blue_voronoi', 'red_voronoi'
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
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    print(f"Launching log_extractor at time: {timestamp}")
    if len(sys.argv) > 1:
        analyze_log(sys.argv[0])
    else:
        #analyze_log("experiment_results/experiment_5rep_600sec/") #braucht aktuell ca 7min... <-no, it was the 50rep one. argv is not working as intended... 40sec ist die Zeit für 5rep
        analyze_log(FOLDER)
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    print(f"log_extractor terminating at time: {timestamp}")