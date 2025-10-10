import os
import re
import ast
import numpy as np
import pandas as pd
from collections import defaultdict

LOG_DIR = "experiment_results"
OUT_DIR = "averaged_tables"
os.makedirs(OUT_DIR, exist_ok=True)

# Regex patterns
RE_REWARD = re.compile(r"reward curve:\s*(\[.*?\])", re.DOTALL)
RE_COLLISIONS = re.compile(r"agent collisions:\s*(\[.*?\])", re.DOTALL)
RE_SCORE = re.compile(r"SCORE:\s*(\[.*?\])", re.DOTALL)
RE_GRABS = re.compile(r"grabs:\s*(\[.*?\])", re.DOTALL)

data = defaultdict(lambda: {
    "reward_curves": [],
    "collisions": [],
    "scores": [],
    "grabs": []
})

def clean_lines(text):
    """Remove warnings and irrelevant lines."""
    lines = text.splitlines()
    cleaned = [
        l for l in lines
        if not l.strip().startswith("Warning!") and
           "Auto-detecting action" not in l and
           not l.strip().startswith("DOES THIS GET CALLED") and
           not l.strip().startswith("solutri")
    ]
    return "\n".join(cleaned)

def get_prefix(filename):
    """Extract prefix before timestamp."""
    parts = filename.split("_")
    return "_".join(parts[:-1])

def safe_eval_array(txt):
    """Convert a numpy-style list '[ 4 43 7 ]' to '[4, 43, 7]'."""
    # Replace multiple spaces between digits with commas
    txt = re.sub(r'(?<=\d)\s+(?=\d)', ', ', txt)
    # Ensure commas between floats too
    txt = re.sub(r'(?<=\d)\s+(?=[\d\-\.])', ', ', txt)
    try:
        return np.array(ast.literal_eval(txt), dtype=float)
    except Exception:
        # try to clean brackets
        txt = txt.strip().replace('\n', '')
        txt = txt.replace('[', '').replace(']', '').strip()
        if txt:
            numbers = [float(x) for x in txt.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
            return np.array(numbers)
        else:
            return np.array([])

for file in os.listdir(LOG_DIR):
    if not file.endswith(".log"):
        continue
    path = os.path.join(LOG_DIR, file)
    with open(path, "r") as f:
        content = clean_lines(f.read())

    prefix = get_prefix(file)

    try:
        reward_match = RE_REWARD.search(content)
        if reward_match:
            txt = reward_match.group(1)
            try:
                rewards = ast.literal_eval(txt)
            except Exception:
                # Try to fix missing commas between dicts
                txt = txt.replace("} {", "}, {")
                rewards = ast.literal_eval(txt)
            agent1 = [entry.get("agent_1", 0.0) for entry in rewards if isinstance(entry, dict)]
            data[prefix]["reward_curves"].append(agent1)

        for key, regex in [("collisions", RE_COLLISIONS),
                           ("scores", RE_SCORE),
                           ("grabs", RE_GRABS)]:
            match = regex.search(content)
            if match:
                arr = safe_eval_array(match.group(1))
                data[prefix][key].append(arr)

    except Exception as e:
        print(f"Error parsing {file}: {e}")

# Group & average in chunks of 5
def chunk(lst, n=5):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

for prefix, vals in data.items():
    for metric, groups in vals.items():
        for i, group in enumerate(chunk(groups, 5)):
            if not group:
                continue
            min_len = min(map(len, group))
            if min_len == 0:
                continue
            group = [g[:min_len] for g in group]
            avg = np.mean(group, axis=0)
            df = pd.DataFrame({
                "index": np.arange(len(avg)),
                f"{metric}_avg": avg
            })
            df.to_csv(f"{OUT_DIR}/{prefix}_{metric}_group{i+1}.csv", index=False)

    # Reward curve separately with "step" and "agent_1_avg_reward"
    for i, group in enumerate(chunk(vals["reward_curves"], 5)):
        if not group:
            continue
        min_len = min(map(len, group))
        if min_len == 0:
            continue
        group = [g[:min_len] for g in group]
        avg = np.mean(group, axis=0)
        df = pd.DataFrame({
            "step": np.arange(len(avg)),
            "agent_1_avg_reward": avg
        })
        df.to_csv(f"{OUT_DIR}/{prefix}_reward_curve_group{i+1}.csv", index=False)

print(" -- Parsing and averaging complete. -- Files saved to:", OUT_DIR)
