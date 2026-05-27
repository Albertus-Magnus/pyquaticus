from multiprocessing import Pool
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from evaluate_q import matrix_to_heatmap

BAGSIZE = 50 # number of episodes to average over for boxplot visualization, default 50
AXISFONT = 22
LEGENDFONT = 17
NUMBERSFONT = 18


# ===== Metric Computation Helper Functions =====

def compute_total_distance(episode_stats, team_agents):
    """
    Compute sum of distances between consecutive positions for a team.
    team_agents: list of agent IDs (e.g., ['agent_0', 'agent_1', 'agent_2'])
    Returns: total distance traveled by all agents in team
    """
    total_dist = 0.0
    for agent_id in team_agents:
        if agent_id in episode_stats['agent_positions']:
            positions = episode_stats['agent_positions'][agent_id]
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    pos1 = np.array(positions[i-1])
                    pos2 = np.array(positions[i])
                    total_dist += np.linalg.norm(pos2 - pos1)
    return total_dist


def compute_area_coverage(episode_stats, team_agents):
    """
    Compute triangle area between 3 agents per frame, averaged across frames.
    Uses shoelace formula for triangle area.
    Returns: average area coverage
    """
    if len(team_agents) != 3:
        return 0.0
    
    areas = []
    positions_by_agent = {agent_id: episode_stats['agent_positions'].get(agent_id, []) for agent_id in team_agents}
    
    # Determine number of frames
    num_frames = min(len(pos) for pos in positions_by_agent.values() if len(pos) > 0)
    if num_frames == 0:
        return 0.0
    
    for frame in range(num_frames):
        try:
            p0 = np.array(positions_by_agent[team_agents[0]][frame])
            p1 = np.array(positions_by_agent[team_agents[1]][frame])
            p2 = np.array(positions_by_agent[team_agents[2]][frame])
            
            # Shoelace formula for triangle area (2D)
            area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
            areas.append(area)
        except (IndexError, TypeError):
            continue

    # normalized by half the map area (6400) because that would be the largest triangle (3 agents assumed)
    return (np.mean(areas) / 6400.) if areas else compute_distance_coverage(episode_stats, team_agents) 


def compute_distance_coverage(episode_stats, team_agents):
    """
    Compute summed pairwise distances between all agents in team, averaged over frames.
    Measure of how well the team is spreading out.
    Returns: average pairwise distance
    normalize by maximum expected distance 180. (more 178,.. but is upper bound estimation anyways)
    """
    if len(team_agents) < 2:
        return 0.0
    
    positions_by_agent = {agent_id: episode_stats['agent_positions'].get(agent_id, []) for agent_id in team_agents}
    
    # Determine number of frames
    num_frames = min(len(pos) for pos in positions_by_agent.values() if len(pos) > 0)
    if num_frames == 0:
        return 0.0
    
    pairwise_distances = []
    for frame in range(num_frames):
        try:
            frame_dist = 0.0
            for i, agent1 in enumerate(team_agents):
                for agent2 in team_agents[i+1:]:
                    pos1 = np.array(positions_by_agent[agent1][frame])
                    pos2 = np.array(positions_by_agent[agent2][frame])
                    frame_dist += np.linalg.norm(pos2 - pos1)
            pairwise_distances.append(frame_dist)
        except (IndexError, TypeError):
            continue
    # 540 = 180 * 3 (max distance and number of pairs for normalization)
    return np.mean(pairwise_distances) / 540. if pairwise_distances else 0.0


def compute_voronoi_coverage(episode_stats, team_agents):
    """
    Compute Voronoi cell uniformity for a team using scipy.spatial.Voronoi.
    Metric: std dev / mean of cell sizes, averaged over frames.
    Lower value = more uniform coverage.
    Returns: average voronoi uniformity metric
    """
    if len(team_agents) != 3:
        return 0.0
    
    positions_by_agent = {agent_id: episode_stats['agent_positions'].get(agent_id, []) for agent_id in team_agents}
    num_frames = min(len(pos) for pos in positions_by_agent.values() if len(pos) > 0)
    if num_frames < 3:
        return 0.0
    
    uniformity_scores = []
    for frame in range(0, num_frames, max(1, num_frames // 10)):  # Sample every 10% of frames to save computation
        try:
            points = []
            for agent_id in team_agents:
                pos = np.array(positions_by_agent[agent_id][frame])
                points.append(pos)
            
            points = np.array(points)
            
            # Add boundary points to avoid infinite regions
            boundary_offset = 100.0
            points_with_boundary = np.vstack([points, [
                [-boundary_offset, -boundary_offset],
                [boundary_offset, -boundary_offset],
                [boundary_offset, boundary_offset],
                [-boundary_offset, boundary_offset]
            ]])
            
            vor = Voronoi(points_with_boundary)
            
            # Compute cell areas for the 3 actual agents (indices 0, 1, 2)
            cell_areas = []
            for agent_idx in range(3):
                region_idx = vor.point_region[agent_idx]
                region = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    vertices = vor.vertices[region]
                    if len(vertices) > 2:
                        # Compute polygon area (shoelace formula)
                        area = 0.5 * abs(sum(vertices[i][0] * vertices[(i+1) % len(vertices)][1] - 
                                             vertices[(i+1) % len(vertices)][0] * vertices[i][1] 
                                             for i in range(len(vertices))))
                        cell_areas.append(area)
            
            if len(cell_areas) == 3 and np.mean(cell_areas) > 0:
                uniformity = np.std(cell_areas) / np.mean(cell_areas) #working as described in report.
                uniformity_scores.append(uniformity)
        except Exception:
            continue
    
    return np.mean(uniformity_scores) if uniformity_scores else 0.0


def compute_defensive_distance(episode_stats, own_team_agents, opponent_team_agents, own_flag_pos_key, opp_flag_pos_key):
    """
    Compute average distance between opponent agent closest to own flag 
    and own agent closest to that opponent.
    Measure of defense effectiveness.
    Returns: average defensive distance
    Normalized by 180 (map diagonal)
    """
    if len(own_team_agents) == 0 or len(opponent_team_agents) == 0:
        return 0.0
    
    own_positions = {agent_id: episode_stats['agent_positions'].get(agent_id, []) for agent_id in own_team_agents}
    opp_positions = {agent_id: episode_stats['agent_positions'].get(agent_id, []) for agent_id in opponent_team_agents}
    own_flag = episode_stats['flag_positions'].get(own_flag_pos_key, [])
    
    num_frames = min(
        min((len(pos) for pos in own_positions.values() if len(pos) > 0), default=0),
        min((len(pos) for pos in opp_positions.values() if len(pos) > 0), default=0),
        len(own_flag)
    )
    if num_frames == 0:
        return 0.0
    
    defensive_distances = []
    for frame in range(num_frames):
        try:
            flag_pos = np.array(own_flag[frame]) if frame < len(own_flag) else None
            if flag_pos is None:
                continue
            
            # Find opponent closest to own flag
            min_opp_dist = float('inf')
            closest_opp_pos = None
            for agent_id in opponent_team_agents:
                if frame < len(opp_positions[agent_id]):
                    opp_pos = np.array(opp_positions[agent_id][frame])
                    dist_to_flag = np.linalg.norm(opp_pos - flag_pos)
                    if dist_to_flag < min_opp_dist:
                        min_opp_dist = dist_to_flag
                        closest_opp_pos = opp_pos
            
            if closest_opp_pos is None:
                continue
            
            # Find own agent closest to that opponent
            min_own_dist = float('inf')
            for agent_id in own_team_agents:
                if frame < len(own_positions[agent_id]):
                    own_pos = np.array(own_positions[agent_id][frame])
                    dist_to_opponent = np.linalg.norm(own_pos - closest_opp_pos)
                    min_own_dist = min(min_own_dist, dist_to_opponent)
            
            if min_own_dist < float('inf'):
                defensive_distances.append(min_own_dist)
        except (IndexError, TypeError):
            continue
    
    return np.mean(defensive_distances) / 180. if defensive_distances else 0.0


def compute_aggressive_distance(episode_stats, own_team_agents, opp_flag_pos_key):
    """
    Compute average distance from own agent closest to opponent flag to opponent flag.
    Measure of attack reach.
    Returns: average aggressive distance
    """
    if len(own_team_agents) == 0:
        return 0.0
    
    own_positions = {agent_id: episode_stats['agent_positions'].get(agent_id, []) for agent_id in own_team_agents}
    opp_flag = episode_stats['flag_positions'].get(opp_flag_pos_key, [])
    
    num_frames = min(
        min((len(pos) for pos in own_positions.values() if len(pos) > 0), default=0),
        len(opp_flag)
    )
    if num_frames == 0:
        return 0.0
    
    aggressive_distances = []
    for frame in range(num_frames):
        try:
            flag_pos = np.array(opp_flag[frame]) if frame < len(opp_flag) else None
            if flag_pos is None:
                continue
            
            # Find own agent closest to opponent flag
            min_dist = float('inf')
            for agent_id in own_team_agents:
                if frame < len(own_positions[agent_id]):
                    own_pos = np.array(own_positions[agent_id][frame])
                    dist = np.linalg.norm(own_pos - flag_pos)
                    min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                aggressive_distances.append(min_dist)
        except (IndexError, TypeError):
            continue
    
    return np.mean(aggressive_distances) / 180. if aggressive_distances else 0.0


def compute_combined_position_score(def_dist, agg_dist, normalize_scale=100.0, wd=1., wo=1.):
    """
    Compute combined position score: 1 - (defensive_distance + aggressive_distance) * 0.5 / normalize_scale
    Higher score = better positioning.
    Returns: combined score (0-1 range approximately)
    """
    combined = (wd * def_dist + wo * agg_dist) * 0.5 # / normalize_scale
    return max(0.0, 1.0 - combined)


def compute_score_tag_ratio(episode_stats):
    """
    Compute score / tags ratio for both teams.
    Represents offensive efficiency.
    Returns: [blue_ratio, red_ratio]
    """
    captures = episode_stats.get('final_score', [0, 0])
    tags = episode_stats.get('tags', [0, 0])
    
    ratios = []
    for i in range(2):
        # Add 1 to avoid division by zero
        if tags[i] == 0:
            ratio = captures[i]  * 2 # Not an expected case, but some reward is given for captures without tags.
            print(f"Warning: No tags for team {i} in score/tag ratio metric.")
        else:
            ratio = captures[i] / (tags[i] + 1)
        ratios.append(ratio)
    
    return ratios


def compute_aggr_def_percentage(episode_stats, team_agents):
    """
    Compute percentage of frames where team agents are on their own side.
    Measure of tactical positioning.
    Returns: percentage (0-100)
    """
    if len(team_agents) == 0:
        print("Warning: No agents in team for aggr_def_percentage metric.")
        return 0.0
    
    on_own_side_data = {agent_id: episode_stats['on_own_side'].get(agent_id, []) for agent_id in team_agents}
    
    num_frames = min(len(data) for data in on_own_side_data.values() if len(data) > 0)
    if num_frames == 0:
        print("Warning: No frame data for on_own_side metric.")
        return 0.0
    
    total_on_own_side = 0
    total_frames = 0
    
    for frame in range(num_frames):
        for agent_id in team_agents:
            if frame < len(on_own_side_data[agent_id]):
                try:
                    on_side = on_own_side_data[agent_id][frame]
                    if on_side:  # Assuming it's boolean True/False
                        total_on_own_side += 1
                    total_frames += 1
                except (IndexError, TypeError):
                    continue
    
    if total_frames == 0:
        print("Warning: No frame data for on_own_side metric.")
        return 0.0
    
    return 100.0 * total_on_own_side / total_frames


def compute_all_metrics(episode_stats):
    """
    Compute all 9 metrics for a single episode.
    Returns: dict with all metric values for blue and red teams
    """
    blue_agents = ['agent_0', 'agent_1', 'agent_2']
    red_agents = ['agent_3', 'agent_4', 'agent_5']
    
    metrics = {
        'total_distance': [
            compute_total_distance(episode_stats, blue_agents),
            compute_total_distance(episode_stats, red_agents)
        ],
        'area_coverage': [
            compute_area_coverage(episode_stats, blue_agents),
            compute_area_coverage(episode_stats, red_agents)
        ],
        'distance_coverage': [
            compute_distance_coverage(episode_stats, blue_agents),
            compute_distance_coverage(episode_stats, red_agents)
        ],
        'voronoi_coverage': [
            compute_voronoi_coverage(episode_stats, blue_agents),
            compute_voronoi_coverage(episode_stats, red_agents)
        ],
        'defensive_distance': [
            compute_defensive_distance(episode_stats, blue_agents, red_agents, 'blue_flag', 'red_flag'),
            compute_defensive_distance(episode_stats, red_agents, blue_agents, 'red_flag', 'blue_flag')
        ],
        'aggressive_distance': [
            compute_aggressive_distance(episode_stats, blue_agents, 'red_flag'),
            compute_aggressive_distance(episode_stats, red_agents, 'blue_flag')
        ],
        'score_tag_ratio': compute_score_tag_ratio(episode_stats),
        'aggr_def_percentage': [
            compute_aggr_def_percentage(episode_stats, blue_agents),
            compute_aggr_def_percentage(episode_stats, red_agents)
        ]
    }
    
    # Compute combined position score from def and agg distances
    metrics['combined_position_score'] = [
        compute_combined_position_score(metrics['defensive_distance'][0], metrics['aggressive_distance'][0]),
        compute_combined_position_score(metrics['defensive_distance'][1], metrics['aggressive_distance'][1])
    ]
    
    return metrics




def evalrender(foldername, parameterset_name):
    # This function is called for each parameter set, it loads the statistics from the specified file and creates visualizations.
    # The visualizations are saved to files in a "figures" subfolder of the parameter set's folder.
    # The name of the output files is based on the parameter set's name and the type of data being visualized (score, etc.).
    
    # Load data from file
    para_file = foldername + parameterset_name + "_eval.npy"
    data = np.load(para_file, allow_pickle=True)

    # print(data)
    # sys.exit()

    # Visualization folder
    vis_folder = foldername + "figures/"
    if not os.path.isdir(vis_folder):
        os.makedirs(vis_folder)

    # Team Blue is 'agent_0', 'agent_1', 'agent_2' and Team Red is 'agent_3', 'agent_4', 'agent_5'
    
    print(f"\n{'='*60}")
    print(f"Computing metrics for {parameterset_name}")
    print(f"Evaluating {len(data)} matches")
    print(f"{'='*60}\n")

    # ===== Computing metrics for all 60 matches =====
    all_metrics = {
        'total_distance': [],
        'area_coverage': [],
        'distance_coverage': [],
        'voronoi_coverage': [],
        'defensive_distance': [],
        'aggressive_distance': [],
        'combined_position_score': [],
        'score_tag_ratio': [],
        'aggr_def_percentage': []
    }
    
    for match_idx, episode_stats in enumerate(data):
        metrics = compute_all_metrics(episode_stats)
        
        for metric_name in all_metrics.keys():
            all_metrics[metric_name].append(metrics[metric_name])
        
        if (match_idx + 1) % 10 == 0:
            print(f"  Processed {match_idx + 1}/{len(data)} matches")
    
    # Convert lists to numpy arrays: shape (num_matches, 2 teams) for each metric
    metric_arrays = {}
    for metric_name, values in all_metrics.items():
        metric_arrays[metric_name] = np.array(values)
    
    print(f"\nMetric computation complete. Computing statistics...\n")

    # ===== Print summary statistics =====
    team_names = ['Team Blue', 'Team Red']
    metric_display_names = {
        'total_distance': 'Total Distance',
        'area_coverage': 'Area Coverage',
        'distance_coverage': 'Distance Coverage',
        'voronoi_coverage': 'Voronoi Uniformity',
        'defensive_distance': 'Defensive Distance',
        'aggressive_distance': 'Aggressive Distance',
        'combined_position_score': 'Combined Position Score',
        'score_tag_ratio': 'Score/Tag Ratio',
        'aggr_def_percentage': 'Aggr-Def Percentage'
    }
    
    print(f"{'Metric':<40} {'Team Blue':<25} {'Team Red':<25}")
    print(f"{'-'*90}")
    
    for metric_name, display_name in metric_display_names.items():
        arr = metric_arrays[metric_name]
        blue_mean = np.mean(arr[:, 0])
        blue_std = np.std(arr[:, 0])
        red_mean = np.mean(arr[:, 1])
        red_std = np.std(arr[:, 1])
        print(f"{display_name:<40} {blue_mean:>8.3f} - {blue_std:<8.3f}   {red_mean:>8.3f} - {red_std:<8.3f}")
    
    print(f"{'-'*90}\n")

    # ===== Generate combined boxplot visualization for all metrics =====
    visualize_all_metrics_combined(metric_arrays, metric_display_names, foldername, parameterset_name)
    
    # ===== Save metrics to file =====
    metrics_file = foldername + parameterset_name + "_metrics.npy"
    np.save(metrics_file, metric_arrays)
    print(f"Metrics saved to: {metrics_file}")
    print(f"Shape: {len(data)} matches x 9 metrics x 2 teams")
    
    print(f"\nVisualizations saved to: {vis_folder}\n")


def group_metrics_by_unit():
    """
    Group metrics by unit type for visualization with appropriate scaling.
    Returns: list of (metric_name, unit_group) tuples ordered for display
    """
    # Metrics grouped by unit type: (metric_name, unit_group, display_name)
    metric_groups = [
        # Total distance metric (group 0)
        ('total_distance', 0, 'Total Distance'),
        # Normalized metrics (group 1) - 0 to 1.0 value range
        ('distance_coverage', 1, 'Distance Coverage'),
        ('defensive_distance', 1, 'Defensive Distance'),
        ('aggressive_distance', 1, 'Aggressive Distance'),
        ('area_coverage', 1, 'Area Coverage'),
        ('combined_position_score', 1, 'Combined Position Score'),
        ('voronoi_coverage', 2, 'Voronoi Uniformity'),
        ('score_tag_ratio', 2, 'Score/Tag Ratio'),
        # Percentage metric (group 2) - 0-100 scale
        ('aggr_def_percentage', 3, 'Aggr-Def Percentage'),
    ]
    return metric_groups


def visualize_all_metrics_combined(metric_arrays, metric_display_names, foldername, parameterset_name):
    """
    Create a single figure with 9 subplots (1x9) showing all metrics as boxplots.
    Metrics are grouped by unit type with separate y-axis scaling per group.
    """
    metric_groups = group_metrics_by_unit()
    
    # Create figure with 9 subplots in 1 row
    fig, axes = plt.subplots(1, 9, figsize=(36, 6))
    
    # Track y-axis ranges for each unit group for dynamic scaling
    y_ranges = {0: [], 1: [], 2: [], 3: []}  # group_id -> list of data for that group
    
    # First pass: collect data for each unit group to determine y-axis ranges
    for metric_name, unit_group, _ in metric_groups:
        arr = metric_arrays[metric_name]
        y_ranges[unit_group].extend(arr[:, 0].tolist())  # Blue team
        y_ranges[unit_group].extend(arr[:, 1].tolist())  # Red team
    
    # Compute y-axis limits for each group
    y_limits = {}
    for group_id in y_ranges:
        if y_ranges[group_id] and group_id != 1:
            data = np.array(y_ranges[group_id])
            min_val = np.min(data)
            max_val = np.max(data)
            margin = (max_val - min_val) * 0.1  # 10% margin
            y_limits[group_id] = (min_val - margin, max_val + margin)
        else:
            y_limits[group_id] = (0, 1)
    
    # Force percentage group (3) to be 0-100
    y_limits[3] = (-5, 105)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    
    # Second pass: create subplots
    for subplot_idx, (metric_name, unit_group, display_name) in enumerate(metric_groups):
        ax = axes[subplot_idx]
        arr = metric_arrays[metric_name]
        
        # Prepare data for boxplot
        data_to_plot = [arr[:, 0], arr[:, 1]]  # Blue and Red teams
        
        # Create boxplot
        bp = ax.boxplot(data_to_plot, labels=['Blue', 'Red'], patch_artist=True, widths=0.5)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Customize appearance
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1)
        for cap in bp['caps']:
            cap.set(linewidth=1)
        
        # Add individual points
        for i, team_data in enumerate(data_to_plot):
            x = np.random.normal(i+1, 0.04, size=len(team_data))
            ax.scatter(x, team_data, alpha=0.25, s=20)
        
        # AXISFONT = 22
        # LEGENDFONT = 17
        # NUMBERSFONT = 18
        NAMEFONT = 25

        # Set axis labels and limits
        ax.set_title(display_name, fontsize=NAMEFONT)
        ax.set_ylim(y_limits[unit_group])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=AXISFONT)
        ax.tick_params(axis='y', labelsize=AXISFONT)
        
        # Add subtle background color to separate unit groups TODO find better colors
        if unit_group == 0:
            ax.set_facecolor('#f0f0f0')  # Light gray for distance metrics
        elif unit_group == 1:
            ax.set_facecolor('#ffffff')  # White for unitless metrics
        else:
            ax.set_facecolor('#f8f8f0')  # Subtle beige for percentage

    # Add title and adjust layout
    # fig.suptitle(f'All Metrics - {parameterset_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = f"{foldername}figures/{parameterset_name}_metrics_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined metrics visualization saved to: {output_file}")


def visualize_metric_boxplot(metric_array, foldername, parameterset_name, metric_name, metric_display_name):
    """
    Create a boxplot visualization for a metric across 60 evaluation matches.
    metric_array: shape (60, 2) where columns are [blue_team, red_team]
    Note: This function is kept for backward compatibility. New code should use visualize_all_metrics_combined().
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for boxplot
    data_to_plot = [metric_array[:, 0], metric_array[:, 1]]  # Blue and Red teams
    
    # Create boxplot
    bp = ax.boxplot(data_to_plot, labels=['Team Blue', 'Team Red'], patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize appearance
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5)
    for cap in bp['caps']:
        cap.set(linewidth=1.5)
    
    ax.set_ylabel(metric_display_name, fontsize=AXISFONT)
    ax.set_title(f"{metric_display_name}\n{parameterset_name}", fontsize=AXISFONT)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add individual points
    for i, team_data in enumerate(data_to_plot):
        x = np.random.normal(i+1, 0.04, size=len(team_data))
        ax.scatter(x, team_data, alpha=0.3, s=30)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{foldername}figures/{parameterset_name}_metric_{metric_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()



def plot_rewards(rewarray, foldername, name):
    """Rewardcurve visualization, rewarray is a list, that contains lists of rewards (per-episode) where each entry consists of 3 entries for the 3 agents of a game.
    Current method is to sum up the 3 agents rewards, removing the last dimension from the above and using a team reward thus.
    """
    # print("SHAPE", np.shape(rewarray)) 
    # print("rewarray example:") #debug print
    # for i in range(len(rewarray)):
    #     print(rewarray[i][500][0])
    
    sum_array = np.sum(np.array(rewarray), 2)
    meandata = np.mean(np.array(sum_array), 0) 
    lowdata = meandata - np.std(sum_array, 0)
    highdata = meandata + np.std(sum_array, 0)
    visualize_reward_curve(meandata, lowdata, highdata, foldername, name)
    visualize_reward_boxplots(sum_array, foldername, name)#TODO test
#End of plot_rewards()


def plot_anythingelse(scorearray, foldername, name, attribute_name):
    """General visualization, scorearray is a list, that contains lists of scores (per-episode) where each entry consists of 2 entries for the 2 teams of a game.
    """
    # Statecount heatmap special case:
    if attribute_name == "Statecount":
        # heatmap is not helpful, too cluttered and labeling clearly is almost impossible with the current version.
        # matrix_to_heatmap(scorearray, attribute_name, foldername + name + "qheatmap.png", attribute_name)
        return

    # The non-boxplot visualization is too cluttered with 1-4000 datapoints, so bags lead to a nice simple visualization, the more sophisticated one is the boxplots.
    scorearray_bagged = average_multiple_runs(scorearray, bagsize=BAGSIZE)
    meandata = np.mean(np.array(scorearray_bagged), 0)
    stddata = np.std(np.array(scorearray_bagged), 0) 
    lowdata = meandata - stddata
    highdata = meandata + stddata
    # I still like the old avg+std visualization curve but I don't need two of the same and boxplots are superior here
    visualize_curve(meandata, lowdata, highdata, foldername, name, attribute_name)
    visualize_curve_boxplots(scorearray, foldername, name, attribute_name, bagsize=BAGSIZE)
    #visualize_many_curves(scorearray, foldername, name, attribute_name)
#End of plot_rewards()


def visualize_curve(meandata, lowdata, highdata, foldername, name, attribute_name):
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    # print("meandata shape:", np.shape(meandata), "lowdata shape:", np.shape(lowdata), "highdata shape:", np.shape(highdata))
    plt.figure(figsize=(12, 6))
    plt.plot([meandata[i][0] for i in range(len(meandata))], label='Team Blue', color='blue', alpha=0.9)
    plt.fill_between(range(len(lowdata)), [lowdata[i][0] for i in range(len(lowdata))], [highdata[i][0] for i in range(len(highdata))], alpha=0.4, color='blue', label='Standard Deviation between training attempts')
    plt.plot([meandata[i][1] for i in range(len(meandata))], label='Team Red', color='red', alpha=0.9)
    plt.fill_between(range(len(lowdata)), [lowdata[i][1] for i in range(len(lowdata))], [highdata[i][1] for i in range(len(highdata))], alpha=0.4, color='red', label='Standard Deviation between training attempts')
    plt.xlabel("Episodes (averaged over each {} episodes)".format(BAGSIZE))
    # Scale x-axis labels to represent actual episodes (multiply by bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[ticks >= 0]  # Remove negative ticks
    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels([f'{int(x*BAGSIZE)}' for x in ticks])
    plt.ylabel(attribute_name)
    plt.title(f"{attribute_name}\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    plt.gca().set_xticks(ticks)
    # plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend(fontsize=12)
    # Save figure to file:
    plt.savefig(f"{foldername}figures/{name}_{attribute_name}.png", dpi=300, bbox_inches='tight') #NOTE changed output folder for different structure: figures and 
    # plt.show()
    plt.close()
#End of visualize_curve()


# def visualize_many_curves(data: np.ndarray, name, ylabel="Score", foldern=f"qtrainlog/folder/"):
def visualize_many_curves(data, foldername, name, attribute_name):
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    # print("meandata shape:", np.shape(data), "lowdata shape:", np.shape(lowdata), "highdata shape:", np.shape(highdata))
    plt.figure(figsize=(12, 6))
    for p in range(len(data)):
        plt.plot([data[p][i][0] for i in range(len(data[p]))], color='blue', alpha=0.6)
    # plt.fill_between(range(len(lowdata)), [lowdata[i][0] for i in range(len(lowdata))], [highdata[i][0] for i in range(len(highdata))], alpha=0.2, color='blue', label='Standard Deviation between training attempts')
    plt.plot([data[p][i][1] for i in range(len(data[p]))], color='red', alpha=0.6)
    # plt.fill_between(range(len(lowdata)), [lowdata[i][1] for i in range(len(lowdata))], [highdata[i][1] for i in range(len(highdata))], alpha=0.2, color='red', label='Standard Deviation between training attempts')
    # plt.xlabel("Episodes (averaged over each {} episodes)".format(BAGSIZE)) #not currently averaged...
    plt.xlabel("Episodes")
    plt.ylabel(attribute_name)
    plt.title(f"{attribute_name}\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    plt.gca().set_xticks(ticks)
    # plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    # plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldername}figures/{name}_{attribute_name}_manyfold.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close() #
#End of visualize_many_curves()


def visualize_reward_curve(meandata, lowdata, highdata, foldername, name):
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    plt.figure(figsize=(12, 6))

    # set range fixed to -120000 to 1000 for comparability
    plt.ylim(-100000, 1000) #TODO adjust?
    
    plt.plot(meandata, label='Mean of team rewards', color='blue')
    plt.fill_between(range(len(lowdata)), lowdata, highdata, alpha=0.2, color='blue', label='Standard Deviation between training attempts')
    # plt.plot(data1, label='Agent 1', color='darkblue')
    # if data2 is not None:
    #     plt.plot(data2, label='Agent 2', color='lightblue')
    plt.xlabel(f"Episodes", fontsize=AXISFONT)
    
    
    #\n(games between qlearning updates)")
    plt.ylabel(f"Reward", fontsize=AXISFONT)#\n(sum of agents per team)")
    #plt.title(f"Rewards during Q-Learn Training\n{name}")
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=NUMBERSFONT)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    #print("\nticks: ",ticks)
    plt.gca().set_xticks(ticks)
    # plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend(fontsize=LEGENDFONT)
    # Save figure to file:
    plt.savefig(f"{foldername}figures/{name}_reward.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close() #
#End of visualize_reward_curve()

def visualize_reward_boxplots(data, foldername, name):  #TODO TODO decide if boxplots are suitable for rewardcurve
    # data is an array of shape (N, T) where N is the number of parallel training attempts and T is the number of episodes, and each entry is the team reward for that episode.
    # We want to create boxplots for every bagsize episodes, showing the distribution of rewards across the N training attempts for that episode range.
    N, T = data.shape
    num_bags = (T + BAGSIZE - 1) // BAGSIZE  # Ceiling division
    averaged_data = np.zeros((N, num_bags))
    for i in range(0, T, BAGSIZE):
        bag_index = i // BAGSIZE
        averaged_data[:, bag_index] = np.mean(data[:, i:i+BAGSIZE], axis=1)
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(averaged_data, positions=np.arange(num_bags), widths=0.8, patch_artist=True)
    
    # Apply blue color scheme
    for box in bp['boxes']:
        box.set(facecolor='lightblue', edgecolor='blue', linewidth=0.6)
    for median in bp['medians']:
        median.set(color='blue', linewidth=1.0)
    for whisker in bp['whiskers']:
        whisker.set(color='blue', linewidth=0.6)
    for cap in bp['caps']:
        cap.set(color='blue', linewidth=0.6)
    
    plt.xlabel(f"Episodes (averaged over each {BAGSIZE} episodes)")
    plt.ylabel(f"Reward (sum of agents per team)")
    plt.title(f"Reward Distribution during Q-Learn Training\n{name}")
    plt.grid(True, alpha=0.3)
    # X ticks: show fewer labels to avoid clutter (every max(1, num_bags//10))
    step = max(1, num_bags // 10)
    indices = np.arange(num_bags)
    plt.gca().set_xticks(indices[::step])
    plt.gca().set_xticklabels([str(i * BAGSIZE) for i in indices[::step]])
    # Save figure to file:
    plt.savefig(f"{foldername}figures/{name}_reward_bxplt.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
#End of visualize_reward_boxplots()


# def load_and_call_helper(name, nrs, folder, ep):
def load_and_call_helper(parameterset):
    from train_qlearn import ParameterSet  # Local import to avoid circular dependency
    name = parameterset.create_name_without_index()
    nrs = parameterset.nrs
    folder = parameterset.foldername
    ep = parameterset.ep

    name_indexed = []
    for i in range(nrs):
        name_indexed.append(name + f"_nr{i}")
    reward_name = [f"{nam}_reward_curve.npy" for nam in name_indexed]
    scores_name = [f"{nam}_scores.npy" for nam in name_indexed]
    tagslist_name = [f"{nam}_tagslist.npy" for nam in name_indexed]
    grabslist_name = [f"{nam}_grabslist.npy" for nam in name_indexed]
    statecount_name = [f"{nam}_statecount.npy" for nam in name_indexed]

    # test names, now redundant
    # reward_name = ["shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_reward_curve.npy", 
    #             "shortpara2_newbool_aggressive_tags_26_hard_lrate0.01_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_reward_curve.npy"]
    # scores_name = ["shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_scores.npy", 
    #             "shortpara2_newbool_aggressive_tags_26_hard_lrate0.01_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_scores.npy"]

    # #  remove below, bugfixing
    # for e in scores_name + reward_name + tagslist_name:
    #     if not os.path.isfile(folder + e):
    #         print(f"Error: File {folder + e} does not exist. Please check the folder and file names.")
    # # : this means all files are available, but the data (column) for ~3 of the parallel training runs must be missing.
    # # Fixing requires making the code more robust to missing values...
    # import sys
    # sys.exit()

    # # DEBUG: Check shapes of all loaded files
    # # print(f"\nChecking file shapes for {name}...")
    # shapes = {}
    # for i in range(len(reward_name)):
    #     try:
    #         data = np.load(folder + reward_name[i])
    #         shapes[i] = data.shape
    #         print(f"  {reward_name[i]}: {data.shape}")
    #     except Exception as e:
    #         print(f"  {reward_name[i]}: ERROR - {e}")

    # # Find mismatched shapes
    # expected_shape = shapes[0]
    # mismatched = [i for i in shapes if shapes[i] != expected_shape]
    # if mismatched:
    #     print(f"\nWARNING: {len(mismatched)} files have different shapes!")
    #     for i in mismatched:
    #         print(f"  Run {i}: {shapes[i]} (expected {expected_shape})")
    """
    Above: utility to detect missing computation logs from training experiments.

    600 out of 1000 episodes terminated:
    defender1_newbool_caps_and_tags_hard_lrate0.1_discount0.9_initq10.0_20nrs_1000ep_no_pre_nr15_reward_curve.npy

    700 out of 1000 episodes terminated:
    pretr2_single_aggressive26_hard_lrate0.15_discount0.95_initq10.0_20nrs_1000ep_500-pretrained_nr10_reward_curve.npy

    Presumeably missing data in a lot of other runs as well... Need to deal with this elegantly.
    3 options:  1) set missing items to average (this effectively removes them but may cause trouble with boxplot metrics?)
                2) remove runs with missing data (in most cases still 19 runs, so acceptable?)
                3) treat runs as shorter (although 600 out of 1000 episodes is not really acceptable)
                4) use the q-table from the last checkpoint of that run(s!) and complete missing data. (maybe later, for now lets use 2))
    """
    # load data from file
    rewardcurve_list = []
    valid_indices = []
    for i in range(len(reward_name)):
        try:
            data = np.load(folder + reward_name[i])
            # print(f"#+#Loaded {reward_name[i]} with shape {data.shape[0]}")
            if data.shape[0] == ep:  # Your expected episode count
                rewardcurve_list.append(data)
                valid_indices.append(i)
        except:
            pass
    # TODO TODO need to document missing values (e.g. batch 8a)
    # print(f"Valid reward curve files: {len(valid_indices)}/{len(reward_name)}")
    rewardcurve = np.array(rewardcurve_list)
    print(f"Loaded {len(valid_indices)}/{len(reward_name)} valid files")
    
    # rewardcurve = np.array([np.load(folder + reward_name[i]) for i in range(len(reward_name))])
    # scorelist = np.array([np.load(folder + scores_name[i]) for i in range(len(scores_name))])
    # tagslist = np.array([np.load(folder + tagslist_name[i]) for i in range(len(tagslist_name))])
    scorelist = np.array([np.load(folder + scores_name[i]) for i in valid_indices])
    tagslist = np.array([np.load(folder + tagslist_name[i]) for i in valid_indices])
    grabslist = np.array([np.load(folder + grabslist_name[i]) for i in valid_indices])
    statecountlist = np.array([np.load(folder + statecount_name[i]) for i in valid_indices])

    #shortening dimensions for readable test prints:    (should be removed after testing)
    # rewardcurve = rewardcurve[:, 5:10, :]#.copy() #copy not necessary?
    # scorelist = scorelist[:, 5:10, :]

    if not os.path.isdir(folder+"figures/"):
        print("Creating folder "+folder+"figures/")
        os.makedirs(folder+"figures/")

    # print(f"visualizing reward curve for {reward_name[0]}...")
    plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])
    # print(f"visualizing scores for {scores_name[0]}...")
    plot_anythingelse(scorelist, folder, scores_name[0][:-(len("_nr0_scores.npy"))], "Score")
    # Blue and red should be swapped for tags specifically
    # print(f"visualizing tags for {tagslist_name[0]}...")
    tagslist = tagslist[:, :, ::-1]  # Swap blue and red teams
    plot_anythingelse(tagslist, folder, tagslist_name[0][:-(len("_nr0_tagslist.npy"))], "Tags")
    # Sometimes we want to know the number of grabs:
    plot_anythingelse(grabslist, folder, grabslist_name[0][:-(len("_nr0_grabslist.npy"))], "Grabs")
    ## plot one state-visit heatmap:
    #plot_anythingelse(statecountlist, folder, grabslist_name[0][:-(len("_statecount.npy"))], "Statecount")
    

    # Compute for every i (0 to 19) the average team score for the last 100 episodes and print it, to get a quick overview of the final performance of the training.
    for i in range(len(scorelist)):
        final_scores = scorelist[i][-100:] # last 100 episodes
        avg_team_score = np.mean(final_scores, axis=0) # average over episodes, resulting in average score for team blue and team red
        print(f"Final average team scores for {name_indexed[i]}: Team Blue: {avg_team_score[0]:.2f}, Team Red: {avg_team_score[1]:.2f}")
    print(f"\nHighest final average team score for {name}: {max(np.mean(scorelist[i][-100:], axis=0)[0] for i in range(len(scorelist))):.2f} (Team Blue), smallest opponent score was {min(np.mean(scorelist[i][-100:], axis=0)[1] for i in range(len(scorelist))):.2f} (Team Red Minimum)")
    

#(circle detection example lists were removed from code)

def circle_detection(positions: np.ndarray) -> bool:
    # Run a check if the trace of positions in example_positions is just going in circles the entire time. If not this should return False.
    
    # Calculate distances between consecutive points
    # distances = np.linalg.norm(np.diff(example_positions, axis=0), axis=1)
    # mean_dist = np.mean(distances)
    # std_dist = np.std(distances)

    # Calculate distances from centroid
    centroid = np.mean(positions, axis=0)
    distances_from_center = np.linalg.norm(positions - centroid, axis=1)
    mean_radius = np.mean(distances_from_center)
    std_radius = np.std(distances_from_center)

    # Check if:
    # 1. Distances between consecutive points are relatively consistent
    # 2. All points are roughly equidistant from centroid (low std relative to mean)
    # 3. The path closes (start and end points are similar)
    #is_circle = (std_dist / mean_dist < 0.2 and 
    is_circle = std_radius / mean_radius < 0.15 #and
                #  np.linalg.norm(example_positions[0] - example_positions[-1]) < mean_dist * 2)

    return is_circle

def average_multiple_runs(data, bagsize=50):
    """
    data is expected to be a np array of shape (N, T, 2) where N is number of runs, 
    T is number of time steps, and the last dimension has size 2 for the two teams.
    This function will compute for a number of runs (T) equal to bagsize the average 
    data value for each team and each parallel run (N) and return this average as an 
    array of shape (N, T / bagsize, 2).
    """
    N, T, num_teams = data.shape
    assert num_teams == 2, "Expected last dimension to be size 2 for two teams"

    num_bags = (T + bagsize - 1) // bagsize  # Ceiling division
    averaged_data = np.zeros((N, num_bags, num_teams))
    
    for i in range(0, T, bagsize):
        bag_index = i // bagsize
        averaged_data[:, bag_index, :] = np.mean(data[:, i:i+bagsize, :], axis=1)
    
    return averaged_data

def visualize_curve_boxplots(fulldata, foldername, name, attribute_name, show_outliers=True, figsize=(12,6), dpi=300, bagsize=50):
    """
    fulldata: numpy array with shape:
      - (N, T, 2) where fulldata[:,t,0] are Team Blue samples at time t and fulldata[:,t,1] Team Red samples
      OR
      - (N, T) where each entry is a scalar for a single team (then treated as Team Blue only)
    foldername, name, attribute_name: same as your original
    show_outliers: whether to show fliers (default False for speed/clarity)
    """
    if bagsize > 1:
        fulldata = average_multiple_runs(fulldata, bagsize=bagsize)

    fulldata = np.asarray(fulldata)
    # print("shape:", fulldata.shape)
    T = fulldata.shape[1]

    # Detect team dimension
    if fulldata.ndim == 3 and fulldata.shape[2] == 2:
        blue_data = [fulldata[:,t,0] for t in range(T)]
        red_data  = [fulldata[:,t,1] for t in range(T)]
    elif fulldata.ndim == 2:
        # Only one team provided: treat as blue
        blue_data = [fulldata[:,t] for t in range(T)]
        red_data = None
    else:
        raise ValueError("fulldata must be shape (N,T,2) or (N,T)")

    fig, ax = plt.subplots(figsize=figsize)

    # positions: for each time t place blue at t-0.2 and red at t+0.2 (adjustable)
    indices = np.arange(T)
    width = 0.5
    pos_blue = indices - width/2
    pos_red  = indices + width/2

    # common boxplot styling
    bp_kwargs = dict(widths=width*0.9, showfliers=show_outliers, patch_artist=True)
    
    # Plot blue boxes
    bp_blue = ax.boxplot(blue_data, positions=pos_blue, **bp_kwargs)
    for box in bp_blue['boxes']:
        box.set(facecolor='lightblue', edgecolor='blue', linewidth=0.6)
    for median in bp_blue['medians']:
        median.set(color='blue', linewidth=1.0)
    for whisker in bp_blue['whiskers']:
        whisker.set(color='blue', linewidth=0.6)
    for cap in bp_blue['caps']:
        cap.set(color='blue', linewidth=0.6)
    for flier in bp_blue['fliers']:
        flier.set(marker='o', markerfacecolor='none', markeredgecolor='blue', markersize=4)

    # Plot red boxes if present
    if red_data is not None:
        bp_red = ax.boxplot(red_data, positions=pos_red, **bp_kwargs)
        for box in bp_red['boxes']:
            box.set(facecolor='mistyrose', edgecolor='red', linewidth=0.6)
        for median in bp_red['medians']:
            median.set(color='red', linewidth=1.0)
        for whisker in bp_red['whiskers']:
            whisker.set(color='red', linewidth=0.6)
        for cap in bp_red['caps']:
            cap.set(color='red', linewidth=0.6)
        for flier in bp_red['fliers']:
            flier.set(marker='o', markerfacecolor='none', markeredgecolor='red', markersize=4)

    # Axis labels and title
    ax.set_xlabel("Episodes", fontsize=AXISFONT)
    if bagsize > 1:
        ax.set_xlabel("Episodes (averaged over each {} episodes)".format(bagsize), fontsize=AXISFONT)
    ax.set_ylabel(attribute_name, fontsize=AXISFONT)
    #ax.set_title(f"{attribute_name}\n{name}", fontsize=AXISFONT)

    # Fixating y-axis to ranges for comparability between plots
    if attribute_name == "Score":
        ax.set_ylim(0, 8)
    elif attribute_name == "Tags":
        ax.set_ylim(0, 8) #TODO adjust?
    elif attribute_name == "Grabs":
        ax.set_ylim(0, 8) #TODO adjust?

    ax.grid(True, alpha=0.3)
    #ax.tick_params(axis='y', labelsize=NUMBERSFONT)
    ax.tick_params(axis='both', labelsize=NUMBERSFONT)

    # X ticks: show fewer labels to avoid clutter (every max(1, T//10))
    step = max(1, T // 10)
    ax.set_xticks(indices[::step])
    ax.set_xticklabels([str(i) for i in indices[::step]])
    if bagsize > 1:
        ax.set_xticklabels([str(i * bagsize) for i in indices[::step]])

    # Legend (create proxies)
    handles = []
    handles.append(plt.Line2D([0],[0], color='blue', lw=3))
    if red_data is not None:
        handles.append(plt.Line2D([0],[0], color='red', lw=3))
        ax.legend(handles, ['Team Blue', 'Team Red'], fontsize=LEGENDFONT)
    else:
        ax.legend(handles, ['Team Blue'], fontsize=LEGENDFONT)

    plt.tight_layout()
    # Ensure target directory exists before saving
    import os
    outdir = os.path.join(foldername, "figures")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}_{attribute_name}_bxplt.png"), dpi=dpi, bbox_inches='tight')
    plt.close()
#End of visualize_curve_boxplots()

if __name__ == "__main__":

    # print("Circledetector test")
    # print(circle_detection(example_positions))
    # print(circle_detection(example2))

    # v1 = np.load("qtrainlog/batch 6c/parameterconfirm18_newbool_aggressive_tags_26_hard_lrate0.15_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_scores.npy")
    # # for i in range(len(v1)):
    #     # print(i,": ", v1[i])
    # # DONE (hardcoded to be random starts) CONCLUSION: All 5 parallel training experiments resulted in the exact same results (qtable, scores, reward, ...), thus rendering 4 out of 5 calculations trash.
    # # WHY is that so? It looks like ignoreseed is set correctly, and it definitely works in principal in the new 26env. 
    
    # # Also, I don't have time to re-run the calculations before the competition deadline, but I can (DONE) rerun it before the meeting (nrs 20)...

    # import sys
    # sys.exit()
    print("~ visualization 2.0 ~")

    #loading in realistic data from newest batch
    # folder = "qtrainlog/batch 6b/"

    names: list[ParameterSet] = []
    i = 0
    ######################################################################
    # # batch 5
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch1", "qtrainlog/batch 5/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sim_speedup=1)) #before this trained with simspeed 10
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch2", "qtrainlog/batch 5/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parametersearch3", "qtrainlog/batch 5/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parametersearch4", "qtrainlog/batch 5/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "parametersearch5", "qtrainlog/batch 5/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False))
    if False: #enable when re-running already generated figures or their score prints
        if False: #these are not necessary to run ever again, too small sample size or ran before important updates to the code (not suited for comparability)
            # # batch 6b
            # #load_and_call_helper("shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre", 1)#_nr0") #alternative way to call the visualization
            # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara1", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=True, sim_speedup=3))
            # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "shortpara2", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "shortpara3", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # # comparison with sharpturns (here non-sharp) #NOTE is nrs 1 useable for me? might be necessary to rerun or just ignore, were useful for finding decent parameters...
            # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara4", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "shortpara5", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "shortpara6", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "shortpara7", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "shortpara8", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "shortpara9", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara10", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "shortpara11", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # # batch 6a
            names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parametersearch17", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parametersearch18", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch9", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch10", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parametersearch11", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parametersearch12", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "parametersearch13", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "parametersearch14", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch15", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch16", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))

            # # batch 6c
            names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # # batch 6d
            # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))

            # # batch 6e    # these look strange, perhaps the timelimit normalized performances? Weird for an entire batch to be like this, [fixed and reran]
            names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6e/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            #if False: 
            names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6e/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
            names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # batch 7a
        # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "previoustest0", "qtrainlog/batch 7a/", 0, boolchange=False, nrs=1, ep=10, teamsize3=True, timelimit=60., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "previoustest1", "qtrainlog/batch 7a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "previoustest2", "qtrainlog/batch 7a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "previoustest3", "qtrainlog/batch 7a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "previoustest4", "qtrainlog/batch 7a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        # batch 7b
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue1", "qtrainlog/batch 7b/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr10_q_table.npy"))
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue2", "qtrainlog/batch 7b/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr8_q_table.npy"))
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue3", "qtrainlog/batch 7b/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 6e/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy"))
        # batch 7c 
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue4", "qtrainlog/batch 7c/", i, boolchange=False, nrs=20, ep=4000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "prevcontinue5", "qtrainlog/batch 7c/", i, boolchange=False, nrs=20, ep=4000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
        names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "prevcontinue6", "qtrainlog/batch 7c/", i, boolchange=False, nrs=20, ep=4000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 6e/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy"))
        # batch 8a
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, True, "pretr1", "qtrainlog/batch 8a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, True, "pretr2", "qtrainlog/batch 8a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        names.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, True, "pretr3", "qtrainlog/batch 8a/", i, boolchange=True, nrs=20, ep=1000)) 
        names.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.99, 10.0, True, "pretr4", "qtrainlog/batch 8a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        names.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, False, "defender1", "qtrainlog/batch 8a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        names.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.99, 10.0, False, "defender2", "qtrainlog/batch 8a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        names.append(ParameterSet("caps_and_tags", "hard", 0.15, 0.95, 10.0, False, "defender3", "qtrainlog/batch 8a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        names.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.99, 10.0, False, "defender4", "qtrainlog/batch 8a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        names.append(ParameterSet("caps_and_tags", "hard", 0.15, 0.95, 10.0, False, "defender5", "qtrainlog/batch 8a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=False))
        # batch 6f (or 6e 3.0)  (rerun because bug...)
        names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6f/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6f/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6f/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6f/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6f/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6f/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
        # batch 6g (control groups)
        names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "boolctrl1", "qtrainlog/batch 6g/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "boolctrl2", "qtrainlog/batch 6g/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6g/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6g/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6g/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6g/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6g/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=True, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6g/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=True, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6g/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=True, sim_speedup=3))
        # batch 6h (getting a confirmation that certain parameters do not result in training success on average; only good parameters were run in breadth so far)
        names.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.85, 10.0, False, "unsuitable_param1", "qtrainlog/batch 6h/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive_rew", "hard", 0.3, 0.8, 10.0, False, "unsuitable_param2", "qtrainlog/batch 6h/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
    if True:
        # batch 6i re-run of wrong reward
        names.append(ParameterSet("single_aggressive26", "hard", 0.2, 0.85, 10.0, False, "unsuitable_param1", "qtrainlog/batch 6h/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.3, 0.8, 10.0, False, "unsuitable_param2", "qtrainlog/batch 6h/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))
        names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "aggrtagsnobool", "qtrainlog/batch 6g/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3))

    #####################################################################
    # for name in names:
    #     load_and_call_helper(name.create_name_without_index(), name.nrs, name.foldername, name.ep)
    
    # Run all scheduled parameters in parallel
    num_jobs = len(names)

    num_workers = max(1, os.cpu_count() - 1)
    # or overwrite with own number:
    #num_workers = 5
    print(f"Rendering plots with {num_workers} workers.")

    with Pool(processes=num_workers) as pool:
        pool.map(load_and_call_helper, names)
    # for nam in names:
    #     load_and_call_helper(nam)




    # testing 
    # plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])
    # plot_anythingelse(scorelist, folder, scores_name[0][:-(len("_nr0_scores.npy"))], "Score")
    # for rew in reward_name:
    #     plot_rewards(np.array([np.load(folder + rew) for rew in reward_name]), folder, rew[:-(len("_nr0_reward_curve.npy"))])
    #     plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])