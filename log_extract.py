# import ast
# import re

# positions = []
# #print("test123")
# with open("match.log") as f:
#     #print("f open")
#     for line in f:
#         print("line:",line)
#         if "obs" not in line:
#             print("not obs")
#             continue
        
#         # extract the dict literal
#         m = re.search(r"obs\s+(.*)", line)
#         if not m:
#             print("not m")
#             continue
        
#         raw = m.group(1)
#         print("raw:",raw)

#         # convert array([...]) → [...]
#         raw = re.sub(r"array\((\[.*?\])\)", r"\1", raw)
#         print("raw converted:",raw)

#         try:
#             obs = ast.literal_eval(raw)
#             print("obs:",obs)
#         except Exception:
#             print("NONE exception")
#             continue

#         pos = obs.get(("agent_0", "pos"))
#         if pos is not None:
#             positions.append(tuple(pos))
#             print("pos:",pos)

# print(positions)


########GOOD REGEX
# import re
# pattern = re.compile(
#     r"\('agent_0', 'pos'\)\s*:\s*array\(\s*\[([^\]]+)\]\s*\)"
# )
# positions = []
# with open("match.log") as f:
#     for line in f:
#         match = pattern.search(line)
#         if match:
#             nums = match.group(1).split(',')
#             pos = tuple(float(x) for x in nums)
#             positions.append(pos)
# print(positions)
##########GOOD REGEX

########GOODPLOT
# import re
# import matplotlib.pyplot as plt

# # --- Extract agent_0 positions from log ---

# pattern = re.compile(
#     r"\('agent_0', 'pos'\)\s*:\s*array\(\s*\[([^\]]+)\]\s*\)"
# )

# positions = []

# with open("match.log", "r") as f:
#     for line in f:
#         match = pattern.search(line)
#         if match:
#             nums = match.group(1).split(',')
#             pos = tuple(float(x) for x in nums)
#             positions.append(pos)

# # --- Prepare path plot ---

# xs = [p[0] for p in positions]
# ys = [p[1] for p in positions]

# plt.figure()
# plt.plot(xs, ys)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Agent 0 Path")
# plt.axis("equal")            # keep aspect ratio correct
# plt.show()
###########GOODPLOT

###########GOODMULTIPLOT
# import re
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # Regex that captures: ('agent_X', 'pos'): array([x, y])
# pattern = re.compile(
#     r"\('agent_(\d+)', 'pos'\)\s*:\s*array\(\s*\[([^\]]+)\]\s*\)"
# )

# # Store per-agent coordinate sequences
# agent_positions = defaultdict(list)

# with open("match.log", "r") as f:
#     for line in f:
#         for match in pattern.finditer(line):
#             agent_id = int(match.group(1))
#             nums = match.group(2).split(',')
#             x, y = float(nums[0]), float(nums[1])
#             agent_positions[agent_id].append((x, y))

# # Plot
# plt.figure()

# for agent_id, coords in sorted(agent_positions.items()):
#     if(agent_id <= 2):
#         continue
#     xs = [p[0] for p in coords]
#     ys = [p[1] for p in coords]
#     plt.plot(xs, ys, label=f"agent_{agent_id}")

# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Agent Paths")
# plt.axis("equal")
# plt.legend()
# plt.tight_layout()
# plt.show()
###############GOODMULTIPLOT


######GOODSCATTERPLOT
# import re
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # Regex for: ('agent_X', 'pos'): array([x, y])
# pattern = re.compile(
#     r"\('agent_(\d+)', 'pos'\)\s*:\s*array\(\s*\[([^\]]+)\]\s*\)"
# )

# agent_positions = defaultdict(list)

# with open("match.log", "r") as f:
#     for line in f:
#         for match in pattern.finditer(line):
#             agent_id = int(match.group(1))
#             nums = match.group(2).split(',')
#             x, y = float(nums[0]), float(nums[1])
#             agent_positions[agent_id].append((x, y))

# plt.figure()

# for agent_id, coords in sorted(agent_positions.items()):
#     if(agent_id <=2): continue
#     xs = [p[0] for p in coords]
#     ys = [p[1] for p in coords]
#     plt.scatter(xs, ys, s=10, label=f"agent_{agent_id}")

# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Agent Paths (Scatter Points)")
# plt.axis("equal")
# plt.legend()
# plt.tight_layout()
# plt.show()
##############GOODSCATTERPLOT

import re
import matplotlib.pyplot as plt # type: ignore
from collections import defaultdict

# Regex patterns
pos_pattern = re.compile(
    r"\('agent_(\d+)', 'pos'\)\s*:\s*array\(\s*\[([^\]]+)\]\s*\)"
)
tag_pattern = re.compile(
    r"\('agent_(\d+)', 'is_tagged'\)\s*:\s*(True|False)"
)

# Track last-known positions and tag states
current_pos = {}                    # agent_id → (x, y)
prev_tag_state = defaultdict(lambda: False)
tag_positions = defaultdict(list)   # agent_id → [(x, y), ...]

with open("match.log", "r") as f:
    for line in f:

        # Position updates
        for m in pos_pattern.finditer(line):
            agent_id = int(m.group(1))
            nums = m.group(2).split(',')
            x, y = float(nums[0]), float(nums[1])
            current_pos[agent_id] = (x, y)

        # Tag updates
        for m in tag_pattern.finditer(line):
            agent_id = int(m.group(1))
            is_tagged = (m.group(2) == "True")

            # Detect False -> True transition
            if (not prev_tag_state[agent_id]) and is_tagged:
                if agent_id in current_pos:
                    tag_positions[agent_id].append(current_pos[agent_id])

            prev_tag_state[agent_id] = is_tagged

# Plotting
plt.figure()

for agent_id, coords in sorted(tag_positions.items()):
    if(agent_id <= 2): continue
    if coords:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        plt.scatter(xs, ys, s=25, label=f"agent_{agent_id}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Tag Event Positions for All Agents")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()
