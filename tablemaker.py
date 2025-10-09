import os
import re
import ast
import numpy as np
import pandas as pd
from collections import defaultdict

# Path to logs
LOG_DIR = "experiment_results"  # adjust if needed

# Regex patterns
RE_REWARD = re.compile(r"reward curve:\s*(\[.*?\])", re.DOTALL)
RE_COLLISIONS = re.compile(r"agent collisions:\s*(\[.*?\])", re.DOTALL)
RE_SCORE = re.compile(r"SCORE:\s*(\[.*?\])", re.DOTALL)
RE_GRABS = re.compile(r"grabs:\s*(\[.*?\])", re.DOTALL)

# Data containers
data = defaultdict(lambda: {
    "reward_curves": [],
    "collisions": [],
    "scores": [],
    "grabs": []
})

def clean_lines(text):
    """Remove warning and auto-detect lines."""
    lines = text.splitlines()
    cleaned = [
        l for l in lines
        if not l.strip().startswith("Warning!") and
           "Auto-detecting action" not in l
    ]
    return "\n".join(cleaned)

def get_prefix(filename):
    """Extract prefix before timestamp, e.g., 'rhea_test_agr'."""
    parts = filename.split("_")
    # remove last element (timestamp)
    return "_".join(parts[:-1])

for file in os.listdir(LOG_DIR):
    if not file.endswith(".log"):
        continue
    with open(os.path.join(LOG_DIR, file), "r") as f:
        content = clean_lines(f.read())
    
    prefix = get_prefix(file)

    # Extract data
    try:
        reward_match = RE_REWARD.search(content)
        if reward_match:
            rewards = ast.literal_eval(reward_match.group(1))
            # Extract agent_1 values only
            agent1 = [entry.get("agent_1", 0.0) for entry in rewards]
            data[prefix]["reward_curves"].append(agent1)

        col_match = RE_COLLISIONS.search(content)
        if col_match:
            collisions = np.array(ast.literal_eval(col_match.group(1)), dtype=float)
            data[prefix]["collisions"].append(collisions)

        score_match = RE_SCORE.search(content)
        if score_match:
            scores = np.array(ast.literal_eval(score_match.group(1)), dtype=float)
            data[prefix]["scores"].append(scores)

        grabs_match = RE_GRABS.search(content)
        if grabs_match:
            grabs = np.array(ast.literal_eval(grabs_match.group(1)), dtype=float)
            data[prefix]["grabs"].append(grabs)

    except Exception as e:
        print(f"Error parsing {file}: {e}")

# ---- Average every 5 logs ----
OUT_DIR = "averaged_tables"
os.makedirs(OUT_DIR, exist_ok=True)

for prefix, vals in data.items():
    # Group in chunks of 5
    def chunk(lst, n=5):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    # Reward Curves (agent_1)
    reward_avg_tables = []
    for i, group in enumerate(chunk(vals["reward_curves"], 5)):
        if not group:
            continue
        min_len = min(map(len, group))
        group = [g[:min_len] for g in group]
        avg = np.mean(group, axis=0)
        df = pd.DataFrame({
            "step": np.arange(len(avg)),
            "agent_1_avg_reward": avg
        })
        df.to_csv(f"{OUT_DIR}/{prefix}_reward_curve_group{i+1}.csv", index=False)
        reward_avg_tables.append(df)

    # Collisions, Score, Grabs
    for metric in ["collisions", "scores", "grabs"]:
        for i, group in enumerate(chunk(vals[metric], 5)):
            if not group:
                continue
            min_len = min(map(len, group))
            group = [g[:min_len] for g in group]
            avg = np.mean(group, axis=0)
            df = pd.DataFrame({
                "index": np.arange(len(avg)),
                f"{metric}_avg": avg
            })
            df.to_csv(f"{OUT_DIR}/{prefix}_{metric}_group{i+1}.csv", index=False)

print("Processing complete! Averaged CSVs saved to", OUT_DIR)
