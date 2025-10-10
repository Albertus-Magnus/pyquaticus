import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG --------
DATA_DIR = "batch combined"
OUT_DIR = "figures"
SAVE_FIGS = True
# batch number
GRP = "3"
# number of steps into the game reward is plotted for
NEM = 300
REDCODE = "#FF0000"
BLUECODE = "#0000FF"

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
    plt.bar(x - width/2, list(team_blue_means.values()), width, color=BLUECODE, alpha=1.0, edgecolor="none", label="Blue Team")
    plt.bar(x + width/2, list(team_red_means.values()), width, color=REDCODE, alpha=1.0, edgecolor="none", label="Red Team")
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
    items = sorted(data[metric].items(), reverse=True)
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
    plt.bar(x - width / 2, team_blue, width, color=BLUECODE, alpha=1.0, edgecolor="none", label=TEAM_LABELS[0])
    plt.bar(x + width / 2, team_red, width, color=REDCODE, alpha=1.0, edgecolor="none", label=TEAM_LABELS[1])
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
    # Use the original name_tag keys (not aliases) to guarantee alignment
    grabs_tags = set(data["grabs"].keys())
    scores_tags = set(data["scores"].keys())
    common_tags = [t for t in sorted(grabs_tags, reverse=True) if t in scores_tags]

    if not common_tags:
        print("No common experiments between grabs and scores; skipping.")
        return

    # collect numeric arrays aligned by common_tags
    grabs_blue_vals = []
    grabs_red_vals = []
    scores_blue_vals = []
    scores_red_vals = []
    display_names = []

    # build display alias but prevent duplicate alias collisions
    alias_counts = {}
    for tag in common_tags:
        disp = alias(tag)
        alias_counts.setdefault(disp, 0)
        alias_counts[disp] += 1
        # if alias appears multiple times, suffix it to keep display unique
        if alias_counts[disp] > 1:
            disp = f"{disp} ({alias_counts[disp]})"
        display_names.append(disp)

        # safe extraction with fallback to nan if missing
        gdf = data["grabs"][tag]
        sdf = data["scores"][tag]

        # grabs: index 0 -> Blue, 1 -> Red
        v_g_blue = float(gdf.loc[gdf["index"] == 0, "grabs_avg"].values[0]) if (gdf["index"] == 0).any() else np.nan
        v_g_red  = float(gdf.loc[gdf["index"] == 1, "grabs_avg"].values[0]) if (gdf["index"] == 1).any() else np.nan
        # scores: index 0 -> Blue, 1 -> Red
        v_s_blue = float(sdf.loc[sdf["index"] == 0, "scores_avg"].values[0]) if (sdf["index"] == 0).any() else np.nan
        v_s_red  = float(sdf.loc[sdf["index"] == 1, "scores_avg"].values[0]) if (sdf["index"] == 1).any() else np.nan

        grabs_blue_vals.append(v_g_blue)
        grabs_red_vals.append(v_g_red)
        scores_blue_vals.append(v_s_blue)
        scores_red_vals.append(v_s_red)

    # Debug print table to ensure perfect alignment
    # print("Plotting the following (display_name, blue_grabs, blue_score, red_grabs, red_score):")
    # for i, tag in enumerate(common_tags):
    #     print(f"{display_names[i]} ({tag}): {grabs_blue_vals[i]:.1f}, {scores_blue_vals[i]:.1f}, "
    #           f"{grabs_red_vals[i]:.1f}, {scores_red_vals[i]:.1f}")

        # Now plot using the aligned arrays
    names = display_names
    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))

    # GRABS bars (left axis)
    bar_g_blue = ax1.bar(x - width/2, grabs_blue_vals, width, label="Blue Grabs",
                        color=BLUECODE, alpha=0.5, edgecolor="none")
    bar_g_red  = ax1.bar(x + width/2, grabs_red_vals,  width, label="Red Grabs",
                        color=REDCODE, alpha=0.5, edgecolor="none")
    ax1.set_ylabel("Average Grabs")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")

    # SCORES bars (right axis) — narrower and offset so they don't fully overlap grabs
    ax2 = ax1.twinx()
    offset = width / 4
    score_width = width #/ 2  # narrower than grabs

    bar_s_blue = ax2.bar(x - offset, scores_blue_vals, width=score_width,
                        label="Blue Score", color=BLUECODE, alpha=1.0, edgecolor="none", linewidth=0.5)
    bar_s_red  = ax2.bar(x + offset, scores_red_vals,  width=score_width,
                        label="Red Score", color=REDCODE, alpha=1.0, edgecolor="none", linewidth=0.5)
    ax2.set_ylabel("Average Score")

    # --- Make both axes share the same visual scale so bars are directly comparable ---
    # compute a sensible common max (ignore NaNs)
    all_vals = np.array([
        v for v in (grabs_blue_vals + grabs_red_vals + scores_blue_vals + scores_red_vals)
        if not (np.isnan(v))
    ])
    if len(all_vals):
        ymax = float(np.nanmax(all_vals)) * 1.10  # 10% headroom
        ax1.set_ylim(0, ymax)
        ax2.set_ylim(0, ymax)

    # Optional: annotate scores for verification
    # def annotate_bars(bars, values, axis):
    #     for rect, val in zip(bars, values):
    #         if np.isnan(val):
    #             continue
    #         h = rect.get_height()
    #         axis.text(rect.get_x() + rect.get_width() / 2, h + ymax * 0.02,
    #                   f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    # annotate_bars(bar_s_blue, scores_blue_vals, ax2)
    # annotate_bars(bar_s_red, scores_red_vals, ax2)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize="small", ncol=2)

    plt.title("Grabs and Scores per Team")
    plt.tight_layout()
    if SAVE_FIGS:
        Path(OUT_DIR).mkdir(exist_ok=True)
        plt.savefig(f"{OUT_DIR}/grabs_scores_combined_group{GRP}.png", dpi=200)
    # plt.show()



plot_grabs_scores_combined()

# -------- Reward curves --------
def plot_reward_subset(substring: str, title_suffix: str):
    subset = {k: v for k, v in data["reward_curve"].items() if substring in k}
    if not subset:
        print(f"No reward_curve files with '{substring}' found.")
        return

    plt.figure(figsize=(10, 6))
    df_all = pd.DataFrame()

    for name_tag, df in sorted(subset.items(), reverse=True):
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
        plt.fill_between(mean.index, mean - std, mean + std, color="none", alpha=0.2)

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