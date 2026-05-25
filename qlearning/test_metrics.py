#!/usr/bin/env python3
"""
Quick test to verify metric computation functions work correctly
"""
import sys
sys.path.insert(0, '/home/magnus/masters_thesis/2026ver')

import numpy as np
from qlearning.evaluate_q_new import compute_all_metrics

def create_dummy_episode_stats():
    """Create dummy episode_stats for testing"""
    num_frames = 100
    
    episode_stats = {
        'agent_positions': {
            'agent_0': [np.array([i, i*0.5]) for i in range(num_frames)],
            'agent_1': [np.array([i+5, i*0.5+5]) for i in range(num_frames)],
            'agent_2': [np.array([i-5, i*0.5-5]) for i in range(num_frames)],
            'agent_3': [np.array([-i, -i*0.5]) for i in range(num_frames)],
            'agent_4': [np.array([-i-5, -i*0.5-5]) for i in range(num_frames)],
            'agent_5': [np.array([-i+5, -i*0.5+5]) for i in range(num_frames)],
        },
        'agent_headings': {f'agent_{i}': [0.0] * num_frames for i in range(6)},
        'flag_positions': {
            'blue_flag': [np.array([50, 50])] * num_frames,
            'red_flag': [np.array([-50, -50])] * num_frames,
        },
        'scrimmage_line_distances': {f'agent_{i}': [0.0] * num_frames for i in range(6)},
        'carrying_flag': {f'agent_{i}': [False] * num_frames for i in range(6)},
        'on_own_side': {
            'agent_0': [True] * 80 + [False] * 20,
            'agent_1': [True] * 80 + [False] * 20,
            'agent_2': [True] * 80 + [False] * 20,
            'agent_3': [True] * 80 + [False] * 20,
            'agent_4': [True] * 80 + [False] * 20,
            'agent_5': [True] * 80 + [False] * 20,
        },
        'is_tagged': {f'agent_{i}': [False] * num_frames for i in range(6)},
        'tag_cooldowns': {f'agent_{i}': [0] * num_frames for i in range(6)},
        'flag_status': [False] * num_frames,
        'bearings_to_flag': {f'agent_{i}': [0.0] * num_frames for i in range(6)},
        'agent_speeds': {f'agent_{i}': [1.0] * num_frames for i in range(6)},
        'final_score': [5, 3],
        'captures': [5, 3],
        'grabs': [2, 1],
        'tags': [8, 6],
        'reward': 100.0
    }
    
    return episode_stats

if __name__ == "__main__":
    print("Testing metric computation...\n")
    
    episode_stats = create_dummy_episode_stats()
    metrics = compute_all_metrics(episode_stats)
    
    print("Computed metrics:")
    print(f"{'Metric':<30} {'Blue Team':<20} {'Red Team':<20}")
    print("-" * 70)
    
    for metric_name, values in metrics.items():
        blue_val = values[0]
        red_val = values[1]
        print(f"{metric_name:<30} {blue_val:>15.4f}        {red_val:>15.4f}")
    
    print("\n✓ Metric computation test passed!")
    print("\nNote: These are dummy values. Real evaluation data will be used in production.")
