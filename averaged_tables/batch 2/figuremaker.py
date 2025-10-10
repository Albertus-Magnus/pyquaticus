import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG --------
DATA_DIR = "."
OUT_DIR = "figures"
SAVE_FIGS = False

# Aliases for prettier labels in plots
ALIASES = {
    "rhea_test_agr":       "Rhea AGR (Hard)",
    "rhea_test_agr_easy":  "Rhea AGR (Easy)",
    "rhea_test_agr_med":   "Rhea AGR (Medium)",
    "rhea_test_cap":       "Rhea CAP (Hard)",
    "rhea_test_cap_easy":  "Rhea CAP (Easy)",
    "rhea_test_cap_med":   "Rhea CAP (Medium)",
    "ultra_def_test":      "UltraDef (Hard)",
    "ultra_def_test_easy": "UltraDef (Easy)",
    "ultra_def_test_med":  "UltraDef (Medium)",
    "mrhea_test_agr_20251010": "MRHEA AGR (Hard)",
    "mrhea_test_agr_easy_20251010": "MRHEA AGR (Easy)",
    "mrhea_test_agr_med_20251010": "MRHEA AGR (Medium)",
    "mrhea_test_cap_20251010": "MRHEA CAP (Hard)",
    "mrhea_test_cap_easy_20251010": "MRHEA CAP (Easy)",
    "mrhea_test_cap_med_20251010": "MRHEA CAP (Medium)"
}

TEAM_LABELS = {0: "Blue Team", 1: "Red Team"}

# number of steps into the game reward is plotted for
NEM = 300

PATTERN = re.compile(
    r"(?P<name>.+)_(?P<metric>collisions|grabs|reward_curve|scores)_group\d+\.csv$",
    re.IGNORECASE
)

# -------- Load files --------
files = list(Path(DATA_DIR).glob("*.csv"))
data = {k: {} for k in ["collisions", "grabs", "reward_curve", "scores"]}
for f in files:
    m = PATTERN.match(f.name)
    if not m:
        continue
    name_tag, metric = m.group("name"), m.group("metric").lower()
    df = pd.read_csv(f)
    data[metric][name_tag] = df

# -------- Helper: alias lookup --------
def alias(name_tag: str) -> str:
    return ALIASES.get(name_tag, name_tag)

# -------- Collisions --------
if data["collisions"]:
    team_blue_means, team_red_means = {}, {}
    
    items = data["collisions"].items()
    items = sorted(items, reverse=True)#data[metric].items())
    for name_tag, df in items: #data["collisions"].items():
        blue = df.loc[df["index"].between(0, 2), "collisions_avg"].mean()
        red = df.loc[df["index"].between(3, 5), "collisions_avg"].mean()
        team_blue_means[alias(name_tag)] = blue
        team_red_means[alias(name_tag)] = red

    names = list(team_blue_means.keys())
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(max(8, len(names) * 0.9), 5))
    plt.bar(x - width/2, list(team_blue_means.values()), width, label="Blue Team")
    plt.bar(x + width/2, list(team_red_means.values()), width, label="Red Team")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Average collisions")
    plt.title("Average Collisions per Team")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        Path(OUT_DIR).mkdir(exist_ok=True)
        plt.savefig(f"{OUT_DIR}/collisions_per_team_group1.png", dpi=200)
    plt.show()
else:
    print("No collisions files found.")

# -------- Generic two-team metric plot --------
def plot_two_team_metric(metric: str, ycol: str, title: str):
    items = sorted(data[metric].items())
    if not items:
        return
    names = [alias(n) for n, _ in items]
    team_blue, team_red = [], []
    for _, df in items:
        v0 = df.loc[df["index"] == 0, ycol].values
        v1 = df.loc[df["index"] == 1, ycol].values
        team_blue.append(float(v0[0]) if len(v0) else np.nan)
        team_red.append(float(v1[0]) if len(v1) else np.nan)
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(max(8, len(names) * 0.9), 5))
    plt.bar(x - width / 2, team_blue, width, label=TEAM_LABELS[0])
    plt.bar(x + width / 2, team_red, width, label=TEAM_LABELS[1])
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel(ycol.replace("_", " "))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        Path(OUT_DIR).mkdir(exist_ok=True)
        plt.savefig(f"{OUT_DIR}/{metric}_group1.png", dpi=200)
    plt.show()

plot_two_team_metric("grabs", "grabs_avg", "Average Grabs per Team")
plot_two_team_metric("scores", "scores_avg", "Average Scores per Team")

# -------- Reward curves --------
def plot_reward_subset(substring: str, title_suffix: str):
    subset = {k: v for k, v in data["reward_curve"].items() if substring in k}
    if not subset:
        print(f"No reward_curve files with '{substring}' found.")
        return

    plt.figure(figsize=(10, 6))
    df_all = pd.DataFrame()

    for name_tag, df in sorted(subset.items()):
        if not {"step", "agent_1_avg_reward"} <= set(df.columns):
            continue
        df_cut = df[df["step"] <= NEM]
        s = df_cut.set_index("step")["agent_1_avg_reward"].rename(alias(name_tag))
        df_all = pd.concat([df_all, s], axis=1)
        plt.plot(s.index, s.values, alpha=0.6, label=alias(name_tag))

    mean = df_all.mean(axis=1)
    std = df_all.std(axis=1)
    if not mean.empty:
        plt.plot(mean.index, mean, lw=2.5, color="k", label="Mean (all runs)")
        plt.fill_between(mean.index, mean - std, mean + std, color="gray", alpha=0.2)

    plt.xlabel(f"Step (≤ {NEM})")
    plt.ylabel("Agent 1 Average Reward")
    plt.title(f"Reward Curves — {title_suffix} (≤ {NEM} steps)")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    if SAVE_FIGS:
        Path(OUT_DIR).mkdir(exist_ok=True)
        plt.savefig(f"{OUT_DIR}/reward_curves_{substring}_group1.png", dpi=200)
    plt.show()


# Two separate figures
plot_reward_subset("agr", "AGR Agents")
plot_reward_subset("cap", "CAP Agents")