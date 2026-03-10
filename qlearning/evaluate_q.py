from matplotlib import pyplot
import numpy as np

FOLDER = "batch 2 hard"
"""
To change batch, change FOLDER here, sometimes "vshard_"+ in vis_helper() beginning, and vis_helper("ddd")-calls in main (very bottom).
"""


def visualize_reward_curve(data0, data1, name, bagsize=50):
    import matplotlib.pyplot as plt
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    plt.figure(figsize=(12, 6))
    plt.plot(data0, label='Agent 0', color='blue')
    plt.plot(data1, label='Agent 1', color='darkblue')
    plt.xlabel(f"Steps (binned by {bagsize} episodes)")
    plt.ylabel(f"Reward (avg per {bagsize} episodes)")
    plt.title(f"Rewards during Q-Learn Training\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in plt.gca().get_xticks()])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"qtrainlog/{FOLDER}/figures/{name}_reward.png", dpi=300, bbox_inches='tight')
    #plt.show()
#End of visualize_reward_curve()

def visualize_curve(data0, data1, ylabel="Score", name="Training Progress", bagsize=50):
    import matplotlib.pyplot as plt
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    plt.figure(figsize=(12, 6))
    plt.plot(data0, label='Team Blue', color='blue')
    plt.plot(data1, label='Team Red', color='red')
    plt.xlabel(f"Steps (binned by {bagsize} episodes)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in plt.gca().get_xticks()])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"qtrainlog/{FOLDER}/figures/{name}_{ylabel}.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close #
#End of visualize_curve()

def matrix_to_heatmap(matrix, title, filename, name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))

    # Reshape multi-dimensional array to 2D for visualization
    matrix_2d = matrix.reshape(64, -1)
    #print(f"matrix_2d shape: {matrix_2d}") #TODO need to redo the reshaping and separate the qtable into (at least) four heatmaps (but shown in one figure?)

    plt.imshow(matrix_2d, cmap='viridis', aspect='auto')
    plt.colorbar(label=name)
    plt.title(name + "\n" + title)
    plt.xlabel('This achsis is separated into the booleans for own flag and enemy flag, as well as the four actions')
    plt.ylabel('This achsis is separated into the 4*4*4=64 positional informations') #TODO figure out how the order of the cells is related to the booleans and pos. inf.'s
    # Save figure to file:
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close() #
#End of matrix_to_heatmap()

def vis_helper(filename_suffix0):
    ############################################################
    filename_suffix0 = "vshard_" + filename_suffix0
    filename_suffix = "qtrainlog/"+FOLDER+"/" + filename_suffix0
    ############################################################

    # Load data for visualization
    #print(f"Loading q-table from file \"{filename_suffix}_q_table.npy\".")
    q_table = np.load(f"{filename_suffix}_q_table.npy")
    s_table = np.load(f"{filename_suffix}_statecount.npy")
    #print("\ns_table:",s_table) 
    s_table = s_table.astype(np.int32)
    s_table = np.abs(s_table) #TODO remove bugfix for negative values in counter-table once not needed
    #print(f"Loading rewardcurve from file \"{filename_suffix}_reward_curve.npy\".")
    rewardcurve = np.load(f"{filename_suffix}_reward_curve.npy")
    #print(f"Loading scorelist from file \"{filename_suffix}_scores.npy\".")
    scorelist = np.load(f"{filename_suffix}_scores.npy")
    #print(scorelist)
    #print(f"Loading grabslist from file \"{filename_suffix}_grabslist.npy\".")
    grabslist = np.load(f"{filename_suffix}_grabslist.npy")
    #print(f"Loading tagslist from file \"{filename_suffix}_tagslist.npy\".")
    tagslist = np.load(f"{filename_suffix}_tagslist.npy")
    # All data loaded

    # Understanding the data shapes and contents
    #print(f"q_table shape: {q_table.shape}")
    #print(f"rewardcurve shape: {rewardcurve.shape}")
    '''rewardcurve is a 2D array of length 3000 (rn 4), where each entry is the 
    reward for agent 0 and agent 1 respectively obtained in that training episode.
    zB [[-1.17930907e+04 -1.13255293e+04]
    [-3.99995513e+03 -3.07597898e+02]
    [-6.07547337e-03 -3.54126305e-02]
    [ 1.71399542e+00 -2.12815341e+02]]'''
    #print(f"scorelist shape: {scorelist.shape}")
    '''same for scorelist (and the following) but entries are integers.
    zB [[ 4 35]
    [ 0 34]
    [ 0 26]
    [ 0 29]]'''
    #print(f"grabslist shape: {grabslist.shape}")
    #print(f"tagslist shape: {tagslist.shape}")
    '''zB [[ 6  4]
    [11  0]
    [15  0]
    [15  0]] (contains episodic entries for env.state['tags'], which 
    contains [{times team blues agents got tagged} {times team reds agents got tagged}])'''
    # print("rewardcurve: ", rewardcurve)
    # print("scorelist: ", scorelist)
    # print("grabslist: ", grabslist)
    # print("tagslist: ", tagslist)

    # Preprocess data where applicable
    # Helper function:
    def avgbags(datalist, bagsize):
        avg_rewards = []
        for i in range(0, len(datalist), bagsize):
            bag = datalist[i:i+bagsize]
            avg_rewards.append(np.mean(bag, axis=0))
        return avg_rewards
    bagsize = 50
    # rewardcurve:
    rewards0 = [step[0] for step in rewardcurve]
    rewards1 = [step[1] for step in rewardcurve]
    rewards0_avg = avgbags(rewards0, bagsize)
    rewards1_avg = avgbags(rewards1, bagsize)
    # score:
    scores0 = [step[0] for step in scorelist]
    scores1 = [step[1] for step in scorelist]
    scores0_avg = avgbags(scores0, bagsize)
    scores1_avg = avgbags(scores1, bagsize)
    # grabs are not necessary, just stored them as file just in case.
    # tags:
    tags0 = [step[0] for step in tagslist] #KEEP IN MIND: tags0 is how many times red agents tagged blue agents, so display reversed as [1 0], not [0 1]...
    tags1 = [step[1] for step in tagslist]
    tags0_avg = avgbags(tags0, bagsize)
    tags1_avg = avgbags(tags1, bagsize)
    # Calculate winrates
    winrate0 = [np.sum(np.array(scores0[i:i+bagsize]) > np.array(scores1[i:i+bagsize])) / bagsize for i in range(0, len(scores0), bagsize)]
    winrate1 = [np.sum(np.array(scores1[i:i+bagsize]) > np.array(scores0[i:i+bagsize])) / bagsize for i in range(0, len(scores1), bagsize)]

    # Visualization
    visualize_reward_curve(rewards0_avg, rewards1_avg, filename_suffix0)
    visualize_curve(scores0_avg, scores1_avg, ylabel=f"Score (avg per {bagsize} episodes)", name=filename_suffix0)
    visualize_curve(tags1_avg, tags0_avg, ylabel=f"Tags (avg per {bagsize} episodes)", name=filename_suffix0)
    matrix_to_heatmap(q_table, filename_suffix0, "qtrainlog/"+FOLDER+"/figures/"+filename_suffix0+"qheatmap.png", "Q-value (expected reward)")
    matrix_to_heatmap(s_table, filename_suffix0, "qtrainlog/"+FOLDER+"/figures/"+filename_suffix0+"visitcount.png", "States visited (count per state)")
    visualize_curve(winrate0, winrate1, ylabel=f"Winrate (avg per {bagsize} episodes)", name=filename_suffix0)

if __name__ == "__main__":

    #filename_suffix0 = "vshard_example_suffix01"
    #filename_suffix = "qtrainlog/vshard_"+filename_suffix
    
    #vis_helper(filename_suffix0)
    print(f"Starting visualization of {FOLDER}.")
    #q_table = np.load("qtrainlog/vshard_example_suffix01_q_table.npy") #TODO visualize Q-Table
    #matrix_to_heatmap(q_table, "test_title", "qtrainlog/vshard_example_suffix01_q_table.png")
    
    # Batch 1 and 2
    vis_helper("lrate0.1_discount0.9_initialq10.0_single_aggressive_rew")
    vis_helper("lrate0.1_discount0.9_initialq10.0_caps_and_grabs")
    vis_helper("lrate0.1_discount0.9_initialq10.0_caps_and_tags")
    vis_helper("lrate0.2_discount0.9_initialq10.0_single_aggressive_rew")
    vis_helper("lrate0.2_discount0.9_initialq10.0_caps_and_grabs")
    vis_helper("lrate0.2_discount0.9_initialq10.0_caps_and_tags")
    vis_helper("lrate0.1_discount0.95_initialq10.0_single_aggressive_rew")
    vis_helper("lrate0.1_discount0.95_initialq10.0_caps_and_grabs")
    vis_helper("lrate0.1_discount0.95_initialq10.0_caps_and_tags")
    vis_helper("lrate0.1_discount0.9_initialq0.0_single_aggressive_rew")
    vis_helper("lrate0.1_discount0.9_initialq0.0_caps_and_tags")

    # Batch 3
    # vis_helper("lrate0.1_discount0.9_initialq10.0_single_aggressive_rew")
    # vis_helper("lrate0.1_discount0.95_initialq10.0_single_aggressive_rew")
    # vis_helper("lrate0.1_discount0.85_initialq10.0_single_aggressive_rew")
    # vis_helper("lrate0.05_discount0.9_initialq10.0_single_aggressive_rew")
    # vis_helper("lrate0.1_discount0.9_initialq10.0_caps_and_tags")
    # vis_helper("lrate0.2_discount0.9_initialq10.0_caps_and_tags")
    # vis_helper("lrate0.2_discount0.95_initialq10.0_caps_and_tags")
    # vis_helper("lrate0.2_discount0.85_initialq10.0_caps_and_tags")
    # vis_helper("lrate0.15_discount0.9_initialq10.0_caps_and_tags")
    # vis_helper("pretrained_pretrained_lrate0.1_discount0.9_initialq10.0_single_aggressive_rew") #TODO pretrained pretrained? sounds dumb
    # vis_helper("pretrained_pretrained_lrate0.1_discount0.9_initialq10.0_caps_and_tags")

    print("Ended visualization.")
    # Print some scores/averages or something for a table for the parameters
    #TODO (or I will just read the maximum avg from the bag50 graph, because overfitting might have occurred at the final of 3000 episodes)