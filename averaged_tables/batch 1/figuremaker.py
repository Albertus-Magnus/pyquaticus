import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG --------
DATA_DIR = "batch 1"
OUT_DIR = "figures"
SAVE_FIGS = False
# batch number
GRP = "1"
# number of steps into the game reward is plotted for
NEM = 300

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
        plt.savefig(f"{OUT_DIR}/collisions_per_team_group{GRP}.png", dpi=200)
    # plt.show()
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
        plt.savefig(f"{OUT_DIR}/{metric}_group{GRP}.png", dpi=200)
    # plt.show()

plot_two_team_metric("grabs", "grabs_avg", "Average Grabs per Team")
plot_two_team_metric("scores", "scores_avg", "Average Scores per Team")

# -------- Combined Grabs and Scores per Team --------
def plot_grabs_scores_combined():
    grabs_items = sorted(data["grabs"].items())
    scores_items = sorted(data["scores"].items())

    if not grabs_items or not scores_items:
        print("Missing grabs or scores files; skipping combined plot.")
        return

    # Collect averages per team for both metrics
    grabs_blue, grabs_red = {}, {}
    scores_blue, scores_red = {}, {}

    for name_tag, df in grabs_items:
        grabs_blue[alias(name_tag)] = float(df.loc[df["index"] == 0, "grabs_avg"].mean())
        grabs_red[alias(name_tag)] = float(df.loc[df["index"] == 1, "grabs_avg"].mean())

    for name_tag, df in scores_items:
        scores_blue[alias(name_tag)] = float(df.loc[df["index"] == 0, "scores_avg"].mean())
        scores_red[alias(name_tag)] = float(df.loc[df["index"] == 1, "scores_avg"].mean())

    #start
    all_names = list(grabs_blue.keys())
    valid_names = [n for n in all_names if n in scores_blue and n in scores_red]

    # Warn if anything is missing
    missing = set(all_names) - set(valid_names)
    if missing:
        print(f"Warning: Skipping {len(missing)} names without both grabs & scores: {missing}")
    for n in valid_names:
        print(f"{n}: Red grabs={grabs_red[n]:.1f}, Red score={scores_red[n]:.1f}")#end


    # Sort by alias order (consistent)
    # names = sorted(grabs_blue.keys())
    # # names_scores = sorted(scores_blue.keys())
    # x = np.arange(len(names))
    names = valid_names
    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))

    # ---- GRABS (left y-axis, bars) ----
    bar_blue = ax1.bar(x - width/2, [grabs_blue[n] for n in names], width, label="Blue Grabs", color="#154360", edgecolor="none", alpha=0.5)
    bar_red  = ax1.bar(x + width/2, [grabs_red[n] for n in names], width, label="Red Grabs", color="#922B21", edgecolor="none", alpha=0.5)
    ax1.set_ylabel("Average Grabs")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    
    # ---- SCORES (right y-axis, bars) ----
    ax2 = ax1.twinx()
    offset = width / 4
    score_width = width #/ 2 
    # bar_blue_score = ax2.bar(x - offset, [scores_blue.get(n, np.nan) for n in names], width=score_width, color="#1f77b4", edgecolor="none", linewidth=2, alpha=1.0, label="Blue Score")
    # bar_red_score = ax2.bar(x + offset, [scores_red.get(n, np.nan) for n in names], width=score_width, color="#d62728", edgecolor="none", linewidth=2, alpha=1.0, label="Red Score")
    bar_blue_score = ax2.bar(x - offset, [scores_blue[n] for n in names], width=score_width, color="#1f77b4", edgecolor="none", linewidth=2, alpha=1.0, label="Blue Score")
    bar_red_score = ax2.bar(x + offset, [scores_red[n] for n in names], width=score_width, color="#d62728", edgecolor="none", linewidth=2, alpha=1.0, label="Red Score")

    ax2.set_ylabel("Average Score")


    # ---- Combined legend ----
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize="small", ncol=2)

    plt.title("Combined Grabs and Scores per Team")
    plt.tight_layout()
    if SAVE_FIGS:
        Path(OUT_DIR).mkdir(exist_ok=True)
        plt.savefig(f"{OUT_DIR}/grabs_scores_combined_group{GRP}.png", dpi=200)
    plt.show()

plot_grabs_scores_combined()

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
        plt.savefig(f"{OUT_DIR}/reward_curves_{substring}_group{GRP}.png", dpi=400)
    # plt.show()


# Two separate figures
plot_reward_subset("agr", "AGR Agents")
plot_reward_subset("cap", "CAP Agents")