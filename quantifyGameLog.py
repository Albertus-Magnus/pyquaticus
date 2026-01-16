import re
from collections import defaultdict
import math
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

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

def analyze_log(log_address):
    # collect all positions for heatmap
    blue_positions = []  # list of (x,y)
    red_positions  = []  # list of (x,y)

    # Track last-known positions and tag states
    current_pos = {}                    # agent_id -> (x, y)
    prev_tag_state = defaultdict(lambda: False) #dictionary because more than one agent is tracked
    #prev_pos = defaultdict(lambda: [0.,0.])
    prev_pos = {}
    tag_positions = defaultdict(list)   # agent_id -> [(x, y), ...]
    bluescore = 0
    redscore = 0
    blue_def_dist = []
    red_def_dist = []
    sum_distances = defaultdict(lambda: 0.)
    for a in blue_agents + red_agents:
        sum_distances[a] = 0. #is a dictionary efficient enough here? ...probably

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
            
            m = flag_pattern.search(line)
            if m:
                #find current flag positions
                blue_nums = m.group(1).split(',')
                red_nums  = m.group(2).split(',')
                blue_cords = (float(blue_nums[0]), float(blue_nums[1]))
                red_cords  = (float(red_nums[0]),  float(red_nums[1]))
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

                    """ min_b_dist = dist(current_pos[closest_to_r], current_pos[min_r_agent]) #set to first distance to be compared (computes twice but should be efficient enough...)
                    for blue_id in blue_agents:
                        dist_to_agent = dist(current_pos[blue_id], current_pos[min_r_agent]) #find the closest blue agent to the "aggressive" or "dangerous" red agent
                        if(dist_to_agent < min_b_dist):
                            min_b_dist = dist_to_agent
                    #now min_b_dist is the distance from the closest blue agent to the most "aggressive" red agent.

                    min_r_dist = dist(current_pos[closest_to_b], current_pos[min_b_agent]) #set to first distance to be compared 
                    for red_id in blue_agents:
                        dist_to_agent = dist(current_pos[red_id], current_pos[min_b_agent]) #find the closest red agent to the "aggressive" or "dangerous" blue agent
                        if(dist_to_agent < min_r_dist):
                            min_r_dist = dist_to_agent      #THIS CODESNIPPET WAS WRONG - probably was confused by the bad naming and convoluded blue vs red logic. TODO Need to do some illustration in my pdf to explain this...
                    #now min_r_dist is the distance from the closest red agent to the most "aggressive" blue agent. """
                    # distance from closest blue to most aggressive red #STARTOF
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
                            min_r_dist = d #ENDOF
                    
                    blue_def_dist.append(min_b_dist)
                    red_def_dist.append(min_r_dist)

            

    # Some last calculations
    bluetags = len(tag_positions[0]) + len(tag_positions[1]) + len(tag_positions[2])
    redtags = len(tag_positions[3]) + len(tag_positions[4]) + len(tag_positions[5])

    # Avoid division by zero
    if bluetags > 0:
        blue_score_per_tag = bluescore / bluetags
    else:
        blue_score_per_tag = 0.0#or float("nan")

    if redtags > 0:
        red_score_per_tag = redscore / redtags
    else:
        red_score_per_tag = 0.0

    blue_def_dist_avg = 0
    for e in blue_def_dist:
        blue_def_dist_avg += e
    #blue_def_dist_avg = blue_def_dist_avg / len(blue_def_dist) #blue_def_dist_avg is the average distance from the closest blue agent to the most "aggressive" red agent.
    blue_def_dist_avg = blue_def_dist_avg / len(blue_def_dist) if len(blue_def_dist) > 0 else 0.0 #blue_def_dist_avg is the average distance from the closest blue agent to the most "aggressive" red agent.
    red_def_dist_avg = 0
    for e in red_def_dist:
        red_def_dist_avg += e
    red_def_dist_avg = red_def_dist_avg / len(red_def_dist) if len(red_def_dist) > 0 else 0.0 #red_def_dist_avg is the average distance from the closest red agent to the most "aggressive" blue agent.

    # Outputting results v2
    print("--- Results: ---")
    print("BLUE Team score/tags (indicates defensive capability of RED agents): ", blue_score_per_tag)
    print("Red Team score/tags (indicates defensive capability of blue agents): ", red_score_per_tag)
    print("Blue Team avg. defensive distance: ", blue_def_dist_avg)
    print("Red Team avg. defensive distance: ", red_def_dist_avg)

    blue_total_dist = 0.0
    red_total_dist = 0.0
    for a in blue_agents + red_agents:
        print(f"Agent {a} distance traveled: {sum_distances[a]}")
        if a in blue_agents:
            blue_total_dist += sum_distances[a]
        elif a in red_agents:
            red_total_dist += sum_distances[a]
    print("Blue Team total distance traveled: ", blue_total_dist)   
    print("Red Team total distance traveled: ", red_total_dist)

    res = dict()
    res['blue_score_per_tag'] = blue_score_per_tag
    res['red_score_per_tag']  = red_score_per_tag
    res['blue_def_avg_dist']  = blue_def_dist_avg
    res['red_def_avg_dist']   = red_def_dist_avg
    #for a in blue_agents + red_agents:
    #    res[f'agent_{a}_dist'] = sum_distances[a]
    res['blue_total_dist'] = blue_total_dist
    res['red_total_dist'] = red_total_dist
    # collect all positions for heatmap
    res['blue_positions'] = blue_positions
    res['red_positions']  = red_positions


    return res

#def graphicmaker(foldername):
    """
    For every file in the given folder we compute the analysis and plot the average of all seeds for every agent type, difficulty and reward choice
    Graphs are plot showing all the averages (labled according to agent type, difficulty, reward choice
    example expected filename: agent_type0_diffmedium_seed7935_reward1.log
    """

def graphicmaker(foldername):
    """v1
    For every file in the given folder we compute the analysis and plot the average of all seeds for every agent type, difficulty and reward choice
    Graphs are plot showing all the averages (labled according to agent type, difficulty, reward choice
    example expected filename: agent_type0_diffmedium_seed7935_reward1.log
    """
    """v2
    For every log file in foldername matching the naming pattern
    agent_type{n}_diff{difficulty}_seed{seed}_reward{r}.log:
      * run analyze_log(file)
      * aggregate results by (agent_type, difficulty, reward)
      * compute averages across seeds for every measured key
      * produce and save plots (PNG) into foldername/analysis_plots/
    Returns a dict: { group_key: averaged_result_dict }
    """

    # filename example: agent_type0_diffmedium_seed7935_reward1.log
    _filename_re = re.compile(
        r"agent_type(?P<atype>\d+)_diff(?P<diff>[^_]+)_seed(?P<seed>\d+)_reward(?P<reward>\d+)\.log$"
    )

    if not os.path.isdir(foldername):
        raise ValueError(f"folder not found: {foldername}")

    groups = defaultdict(list)  # (atype, diff, reward) -> list of result dicts
    skipped = []

    for fn in sorted(os.listdir(foldername)):
        m = _filename_re.match(fn)
        if not m:
            # skip non-matching files
            continue
        atype = int(m.group("atype"))
        diff = m.group("diff")
        reward = int(m.group("reward"))
        filepath = os.path.join(foldername, fn)
        try:
            res = analyze_log(filepath)
        except Exception as e:
            skipped.append((fn, str(e)))
            continue
        groups[(atype, diff, reward)].append(res)

    if not groups:
        raise RuntimeError("No matching log files found or no results produced.")

    # Compute averages for each group
    averaged = dict()
    # Also store concatenated positions per team
    group_positions = dict()  # key -> {"blue": [...], "red": [...]}

    # collect full set of agent keys to plot consistently
    #all_agent_keys = set()
    all_metric_keys = set()
    for key, res_list in groups.items():
        # collect metric keys
        for r in res_list:
            #all_agent_keys.update(k for k in r.keys() if k.startswith("agent_") and k.endswith("_dist"))
            all_metric_keys.update(r.keys())

    # average aggregation
    for key, res_list in groups.items():
        avg = dict()
        # gather all metrics present in at least one result
        metrics = set().union(*[set(r.keys()) for r in res_list])
        '''for metric in metrics:
            vals = [r[metric] for r in res_list if metric in r and r[metric] is not None]
            if vals:
                # numeric average
                try:
                    avg_val = float(sum(vals)) / len(vals)
                except Exception:
                    # fallback to statistics.mean for robustness
                    avg_val = statistics.mean(vals)
            else:
                avg_val = None
            avg[metric] = avg_val'''
        for metric in metrics:
            vals = [r[metric] for r in res_list if metric in r and r[metric] is not None]

            if not vals:
                avg[metric] = None
                continue

            # Only average numeric values
            if isinstance(vals[0], (int, float)):
                avg[metric] = float(sum(vals)) / len(vals)
            else:
                # Non-numeric (e.g., position lists) -> do not average
                avg[metric] = None

        averaged[key] = avg
        # get concatenated positions for heatmap
        all_blue = []
        all_red  = []
        for r in res_list:
            all_blue.extend(r.get("blue_positions", []))
            all_red.extend(r.get("red_positions", []))

        group_positions[key] = {
            "blue": all_blue,
            "red":  all_red
        }


    # Canonical sort order
    DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}

    def sort_key(k):
        atype, diff, reward = k
        return (atype, DIFF_ORDER.get(diff, 99), reward)


    # Prepare HELPERS for plotting
    def plot_pairs(metric_blue, metric_red, title, ylabel, filename):
        blue_vals = [averaged[k].get(metric_blue, np.nan) for k in sorted_keys]
        red_vals  = [averaged[k].get(metric_red,  np.nan) for k in sorted_keys]

        x = np.arange(len(sorted_keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(x)*0.7), 5))

        ax.bar(x - width/2, blue_vals, width, label="Blue", color="tab:blue")
        ax.bar(x + width/2, red_vals,  width, label="Red",  color="tab:red")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

        # Add separators and group headers
        add_group_separators(ax, sorted_keys, labels)

        plt.tight_layout()
        outfile = os.path.join(plotdir, filename)
        plt.savefig(outfile)
        plt.close(fig)
        
    def add_group_separators(ax, keys, labels):
        # Draw vertical lines when agent_type changes
        last_type = None
        for i, k in enumerate(keys):
            atype = k[0]
            if last_type is None:
                last_type = atype
                continue
            if atype != last_type:
                ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
                last_type = atype

        # Add big group labels on top
        centers = {}
        for i, k in enumerate(keys):
            centers.setdefault(k[0], []).append(i)

        ymin, ymax = ax.get_ylim()
        y = ymax * 1.02
        for atype, idxs in centers.items():
            c = sum(idxs) / len(idxs)
            ax.text(c, y, f"agent_type {atype}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    def plot_heatmap(positions, title, filename, bins=50):
        if not positions:
            return

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        fig, ax = plt.subplots(figsize=(6, 5))

        h = ax.hist2d(xs, ys, bins=bins, cmap="hot")
        plt.colorbar(h[3], ax=ax)

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()
        plt.savefig(os.path.join(plotdir, filename))
        plt.close(fig)

    # END OF HELPERS


    # prepare plotting directory
    plotdir = os.path.join(foldername, "analysis_plots")
    os.makedirs(plotdir, exist_ok=True)

    # prepare labels and sorted order for consistent plots
    #sorted_keys = sorted(averaged.keys(), key=lambda k: (k[0], k[1], k[2]))
    sorted_keys = sorted(averaged.keys(), key=sort_key)
    #labels = [f"type{k[0]}_diff{k[1]}_r{k[2]}" for k in sorted_keys]
    labels = [f"type{k[0]}_{k[1]}_r{k[2]}" for k in sorted_keys]


    # Team score per tag (blue_score_per_tag, red_score_per_tag)
    plot_pairs(
        "blue_score_per_tag", "red_score_per_tag",
        title="Score per Tag (Blue vs Red)",
        ylabel="score / tag",
        filename="score_per_tag_blue_vs_red.png"
    )

    # Team defensive average distances
    plot_pairs(
        "blue_def_avg_dist", "red_def_avg_dist",
        title="Defensive Avg Distance (Blue vs Red)",
        ylabel="distance",
        filename="def_avg_dist_blue_vs_red.png"
    )

    # Team total distances (instead of Per-agent distances plot)
    plot_pairs(
        "blue_total_dist", "red_total_dist",
        title="Total Distance Traveled (Blue vs Red)",
        ylabel="distance",
        filename="total_dist_blue_vs_red.png"
    )

    # Heatmaps per experiment configuration
    for key in sorted_keys:
        atype, diff, reward = key
        label = f"type{atype}_{diff}_r{reward}"

        pos = group_positions[key]

        plot_heatmap(
            pos["blue"],
            title=f"Blue heatmap — {label}",
            filename=f"heatmap_blue_{label}.png"
        )

        plot_heatmap(
            pos["red"],
            title=f"Red heatmap — {label}",
            filename=f"heatmap_red_{label}.png"
        )


    

    # Save a small summary CSV with averaged numbers
    import csv
    csvfile = os.path.join(plotdir, "averaged_results.csv")
    # columns = group_key fields + sorted metric keys
    metric_columns = sorted(list(all_metric_keys))
    with open(csvfile, "w", newline="") as cf:
        writer = csv.writer(cf)
        header = ["agent_type", "difficulty", "reward"] + metric_columns
        writer.writerow(header)
        for key in sorted_keys:
            atype, diff, reward = key
            row = [atype, diff, reward]
            for mc in metric_columns:
                val = averaged[key].get(mc, "")
                row.append("" if val is None else val)
            writer.writerow(row)

    # Optionally print summary of what was done (kept minimal)
    if skipped:
        print(f"Skipped {len(skipped)} files due to errors. See console for details.")
        for fn, err in skipped:
            print(f" - {fn}: {err}")

    return averaged


##############################
if __name__ == "__main__":
    #analyze_log("match.log")
    graphicmaker("experiment_results/experiment_20260115_215621/")
##############################
