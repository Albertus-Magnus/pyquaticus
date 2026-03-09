from matplotlib import pyplot
import numpy as np


filename_suffix = "vshard_example_suffix01"
#filename_suffix = "qtrainlog/vshard_"+filename_suffix
filename_suffix = "qtrainlog/"+filename_suffix

# Load data for visualization
print(f"Loading q-table from file \"{filename_suffix}_q_table.npy\".")
q_table = np.load(f"{filename_suffix}_q_table.npy")
print(f"Loading rewardcurve from file \"{filename_suffix}_reward_curve.npy\".")
rewardcurve = np.load(f"{filename_suffix}_reward_curve.npy")
print(f"Loading scorelist from file \"{filename_suffix}_scores.npy\".")
scorelist = np.load(f"{filename_suffix}_scores.npy")
print(f"Loading grabslist from file \"{filename_suffix}_grabslist.npy\".")
grabslist = np.load(f"{filename_suffix}_grabslist.npy")
print(f"Loading tagslist from file \"{filename_suffix}_tagslist.npy\".")
tagslist = np.load(f"{filename_suffix}_tagslist.npy")
# All data loaded

# Understanding the data shapes and contents
print(f"q_table shape: {q_table.shape}")
print(f"rewardcurve shape: {rewardcurve.shape}")
'''rewardcurve is a 2D array of length 3000 (rn 4), where each entry is the 
reward for agent 0 and agent 1 respectively obtained in that training episode.
zB [[-1.17930907e+04 -1.13255293e+04]
 [-3.99995513e+03 -3.07597898e+02]
 [-6.07547337e-03 -3.54126305e-02]
 [ 1.71399542e+00 -2.12815341e+02]]'''
print(f"scorelist shape: {scorelist.shape}")
'''same for scorelist (and the following) but entries are integers.
zB [[ 4 35]
 [ 0 34]
 [ 0 26]
 [ 0 29]]'''
print(f"grabslist shape: {grabslist.shape}")
print(f"tagslist shape: {tagslist.shape}")
'''zB [[ 6  4]
 [11  0]
 [15  0]
 [15  0]] (contains episodic entries for env.state['tags'], which 
 contains [{times team blues agents got tagged} {times team reds agents got tagged}])'''
print("rewardcurve: ", rewardcurve)
# print("scorelist: ", scorelist)
# print("grabslist: ", grabslist)
# print("tagslist: ", tagslist)

# Preprocess data where applicable
rewards0 = [step[0] for step in rewardcurve]
rewards1 = [step[1] for step in rewardcurve]
def avgbags(datalist, bagsize):
    avg_rewards = []
    for i in range(0, len(datalist), bagsize):
        bag = datalist[i:i+bagsize]
        avg_rewards.append(np.mean(bag, axis=0))
    return avg_rewards
rewards0_avg = avgbags(rewards0, 2)
rewards1_avg = avgbags(rewards1, 2)

# Visualization
def visualize_reward_curve(data0, data1):
    import matplotlib.pyplot as plt
    #reward_curve = np.load(reward_curve_file, allow_pickle=True)
    #rewards0 = [step[0] for step in reward_curve]
    #rewards1 = [step[1] for step in reward_curve]
    plt.figure(figsize=(12, 6))
    plt.plot(data0, label='Agent 0 Rewards')
    plt.plot(data1, label='Agent 1 Rewards')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.show()
# Plot average rewards
#pyplot.plot(avg_rewards, label='Average Rewards', linewidth=2)
#End of visualize_reward_curve()
visualize_reward_curve(rewards0_avg, rewards1_avg)

# Print some scores/averages or something for a table for the parameters
#TODO