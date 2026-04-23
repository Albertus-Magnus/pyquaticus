# import sys
import os

from matplotlib import pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
from train_qlearn import ParameterSet

FOLDER = "batch 9"#zb "batch 5"
"""
To change batch, change FOLDER here, sometimes "vshard_"+ in vis_helper() beginning (now automated), and vis_helper("ddd")-calls in main (very bottom).
"""


# def visualize_reward_curve(data0, data1, name, bagsize=50, foldern=f"qtrainlog/{FOLDER}/"):
#     # reward_curve = np.load(reward_curve_file, allow_pickle=True)
#     # rewards0 = [step[0] for step in reward_curve]
#     # rewards1 = [step[1] for step in reward_curve]
#     plt.figure(figsize=(12, 6))
#     plt.plot(data0, label='Agent 0', color='blue')
#     plt.plot(data1, label='Agent 1', color='darkblue')
#     plt.xlabel(f"Steps (binned by {bagsize} episodes)")
#     plt.ylabel(f"Reward (avg per {bagsize} episodes)")
#     plt.title(f"Rewards during Q-Learn Training\n{name}")
#     plt.grid(True)
#     # Scale x-achsis labels times 50 (or times bagsize):
#     ticks = plt.gca().get_xticks()
#     ticks = ticks[1:(len(ticks)-1)]
#     #print("\nticks: ",ticks)
#     plt.gca().set_xticks(ticks)
#     plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
#     plt.legend()
#     # Save figure to file:
#     plt.savefig(f"{foldern}figures/{name}_reward.png", dpi=300, bbox_inches='tight')
#     #plt.show()
#     plt.close() #
# #End of visualize_reward_curve()

def visualize_reward_curve(data0, data1, name, bagsize=50, foldern=f"qtrainlog/{FOLDER}/", data2=None):
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    plt.figure(figsize=(12, 6))
    plt.plot(data0, label='Agent 0', color='blue')
    plt.plot(data1, label='Agent 1', color='darkblue')
    if data2 is not None:
        plt.plot(data2, label='Agent 2', color='lightblue')
    plt.xlabel(f"Steps (binned by {bagsize} episodes)")
    plt.ylabel(f"Reward (avg per {bagsize} episodes)")
    plt.title(f"Rewards during Q-Learn Training (3v3)\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    #print("\nticks: ",ticks)
    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldern}figures/{name}_reward.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close() #
#End of visualize_reward_curve()

def visualize_reward_quick_and_ugly(data0, data1, data2, name, bagsize=1, foldern="qtrainlog/batch 3/quickfigures/"):#"enhancegrablong_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_i499"):
    """
    foldern example
    enhancegrablong_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre 
    (eventuell mit _i499 ?)
    """
    foldern = "qtrainlog/batch 3/quickfigures/" #+ name
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    plt.figure(figsize=(12, 6))
    plt.plot(data0, label='Agent 0', color='blue')
    plt.plot(data1, label='Agent 1', color='darkblue')
    plt.plot(data2, label='Agent 2', color='lightblue')
    plt.xlabel(f"Steps (binned by {bagsize} episodes)")
    plt.ylabel(f"Reward (avg per {bagsize} episodes)")
    plt.title(f"Rewards during Q-Learn Training (3v3)\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    #print("\nticks: ",ticks)
    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldern}{name}_reward.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close() #
#End of visualize_reward_quick_and_ugly()

def visualize_curve_quickandugly(data0, data1, ylabel="Score", name="Training Progress", bagsize=1, foldern="qtrainlog/batch 3/quickfigures/"):
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
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldern}{name}_{ylabel}.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close #
#End of visualize_curve_quickandugly()

def visualize_curve(data0, data1, ylabel="Score", name="Training Progress", bagsize=50, foldern=f"qtrainlog/{FOLDER}/"):
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
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldern}figures/{name}_{ylabel}.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close #
#End of visualize_curve()


def matrix_to_heatmap(matrix, title, filename, name): 
    # Great visualization and labeling now (finally), TODO need to set it to render all heatmaps without further input. Also TODO probably rename this method and un-comment the original (most above) one...
    import numpy as np
    import matplotlib.pyplot as plt

    # Shape: (a, b, c, d, e, f)
    a, b, c, d, e, f = matrix.shape

    row_labels = [f"{i},{j},{k}" for i in range(a) for j in range(b) for k in range(c)]
    col_labels = [f"{i},{j}" for i in range(d) for j in range(e)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for f_idx in range(f):
        ax = axes[f_idx]

        # Fix f dimension, reshape remaining
        submatrix = matrix[..., f_idx]  # shape: (a,b,c,d,e)
        submatrix_2d = submatrix.reshape(a*b*c, d*e)

        im = ax.imshow(submatrix_2d, cmap='viridis', aspect='auto')

        ax.set_title(f"{name} | action={f_idx}")

        # Optional: only show a few ticks (otherwise it's cluttered)
        ax.set_xticks(range(d*e))
        ax.set_xticklabels(col_labels, rotation=45, fontsize=8)

        ax.set_yticks(range(0, len(row_labels), 8))  # reduce density
        ax.set_yticklabels(row_labels[::8], fontsize=8)

        ax.set_xlabel("(own_flag grabbed, opp_flag grabbed)")
        ax.set_ylabel("(objective, opp1, opp2)")

    fig.suptitle(name + "\n" + title)

    # Shared colorbar
    fig.colorbar(im, ax=axes, label=name)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
#End of matrix_to_heatmap()

def vis_helper(filename_suffix0):
    #TWENTY = 20
    """Used for batches 1-5 to create the visualizations. Has to be called for every training run (per filename-prefix)."""
    ############################################################
    #if FOLDER=="batch 9": filename_suffix0 = "vshard_" + filename_suffix0 #this is wrong now, was just there to deal with batch 5(or 4?)
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
    ####################################################################################################################################################
    visualize_reward_curve(rewards0_avg, rewards1_avg, filename_suffix0)
    visualize_curve(scores0_avg, scores1_avg, ylabel=f"Score (avg per {bagsize} episodes)", name=filename_suffix0)
    visualize_curve(tags1_avg, tags0_avg, ylabel=f"Tags (avg per {bagsize} episodes)", name=filename_suffix0)
    matrix_to_heatmap(q_table, filename_suffix0, "qtrainlog/"+FOLDER+"/figures/"+filename_suffix0+"qheatmap.png", "Q-value (expected reward)")
    # matrix_to_heatmap(s_table, filename_suffix0, "qtrainlog/"+FOLDER+"/figures/"+filename_suffix0+"visitcount.png", "States visited (count per state)")
    # TODO reactivate heatmap for statecount, but need to deal with different dimensionality (see resulting error...)
    visualize_curve(winrate0, winrate1, ylabel=f"Winrate (avg per {bagsize} episodes)", name=filename_suffix0)
    ####################################################################################################################################################
    print(f"{filename_suffix0} 20th value - Rewards: {rewards0_avg[len(rewards0_avg) - 1]}, {rewards1_avg[len(rewards0_avg) - 1]}") # 9 for tenth step, TWENTY - 1 for 20th step (500 vs 1000 episodes)
    print(f"{filename_suffix0} 20th value - Scores: {scores0_avg[len(rewards0_avg) - 1]}, {scores1_avg[len(rewards0_avg) - 1]}")
    print(f"{filename_suffix0} 20th value - Winrate: {winrate0[len(rewards0_avg) - 1]}, {winrate1[len(rewards0_avg) - 1]}")
    print(f"{filename_suffix0} 20th value - Tags: {tags0_avg[len(rewards0_avg) - 1]}, {tags1_avg[len(rewards0_avg) - 1]}")

def avg_vis_helper(paraset: ParameterSet):
    TWENTY = paraset.nrs #used to be 20
    """Is called for every group of training runs using the same parameters to generate the visualizations of averages between the same parameters (e.g. 20 training runs into one plot)."""
    filename_suffix0 = paraset.create_name_without_index()
    ############################################################
    #if FOLDER=="batch 2 hard": filename_suffix0 = "vshard_" + filename_suffix0 #not necessary for avg_vis_helper
    filename_suffix = paraset.foldername + filename_suffix0
    ############################################################

    # Load data for visualization #TODO testing with scores rn, need to reenable the others
    q_table = np.load(f"{filename_suffix}_nr0_q_table.npy")

    rewardcurve = [np.load(f"{filename_suffix}_nr{i}_reward_curve.npy") for i in range(TWENTY)]
    scorelist = [np.load(f"{filename_suffix}_nr{i}_scores.npy") for i in range(TWENTY)]
    grabslist = [np.load(f"{filename_suffix}_nr{i}_grabslist.npy") for i in range(TWENTY)]
    tagslist = [np.load(f"{filename_suffix}_nr{i}_tagslist.npy") for i in range(TWENTY)]
    # All data loaded
    # print the four above lists
    # print(f"rewardcurve: {len(rewardcurve)}")
    # print(f"scorelist: {scorelist}")
    # print(f"length={len(scorelist)}")
    # print(f"grabslist: {len(grabslist)}")
    # print(f"tagslist: {len(tagslist)}")

    # Compute averages of (zB) 20 training runs and (for now) process like the previous non-averaged per-episode lists
    def avg_twoteamlist(list_of_twos):
        list_avg = []
        # Presumes all 20 runs have the same length.
        for i in range(len(list_of_twos[0])): 
            s0 = 0.
            s1 = 0.
            for z in range(len(list_of_twos)):
                s0 += list_of_twos[z][i][0]
                s1 += list_of_twos[z][i][1]
                # lenght expected to be 2 is indeed 2
            s0 = s0 / len(list_of_twos)
            s1 = s1 / len(list_of_twos)
            list_avg.append([s0, s1])
        return list_avg
    
    # Compute averages of (zB) 20 training runs and (for now) process like the previous non-averaged per-episode lists
    # Variant for 3v3 (three agents)
    def avg_threeteamlist(list_of_threes):
        list_avg = []
        # Presumes all 20 runs have the same length.
        for i in range(len(list_of_threes[0])): 
            s0 = 0.
            s1 = 0.
            s2 = 0.
            for z in range(len(list_of_threes)):
                s0 += list_of_threes[z][i][0]
                s1 += list_of_threes[z][i][1]
                s2 += list_of_threes[z][i][2]
                # lenght expected to be 3 now
            
            s0 = s0 / len(list_of_threes)
            s1 = s1 / len(list_of_threes)
            s2 = s2 / len(list_of_threes)
            list_avg.append([s0, s1, s2])
        return list_avg
    
    if paraset.teamsize3:
        rewardcurve_avg = avg_threeteamlist(rewardcurve)
    else:
        rewardcurve_avg = avg_twoteamlist(rewardcurve)
    scorelist_avg = avg_twoteamlist(scorelist)
    grabslist_avg = avg_twoteamlist(grabslist)
    tagslist_avg = avg_twoteamlist(tagslist)
    # All averages computed


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
    rewards0 = [step[0] for step in rewardcurve_avg]
    rewards1 = [step[1] for step in rewardcurve_avg]
    if paraset.teamsize3:
        rewards2 = [step[2] for step in rewardcurve_avg]
    rewards0_avg = avgbags(rewards0, bagsize)
    rewards1_avg = avgbags(rewards1, bagsize)
    # print(f"rewards0 length is {len(rewards0)}, rewards0avg length is {len(rewards0_avg)}")
    # score:
    #scores0 = [step[0] for step in scorelist]#below line is new this line
    scores0 = [step[0] for step in scorelist_avg]
    # scores1 = [step[1] for step in scorelist]#below line is new this line
    scores1 = [step[1] for step in scorelist_avg]
    scores0_avg = avgbags(scores0, bagsize)
    scores1_avg = avgbags(scores1, bagsize)
    # grabs are not necessary, just stored them as file just in case.
    # tags:
    tags0 = [step[0] for step in tagslist_avg] #KEEP IN MIND: tags0 is how many times red agents tagged blue agents, so display reversed as [1 0], not [0 1]...
    tags1 = [step[1] for step in tagslist_avg]
    tags0_avg = avgbags(tags0, bagsize)
    tags1_avg = avgbags(tags1, bagsize)
    # Calculate winrates
    winrate0 = [np.sum(np.array(scores0[i:i+bagsize]) > np.array(scores1[i:i+bagsize])) / bagsize for i in range(0, len(scores0), bagsize)]
    winrate1 = [np.sum(np.array(scores1[i:i+bagsize]) > np.array(scores0[i:i+bagsize])) / bagsize for i in range(0, len(scores1), bagsize)]

    # Visualization
    ####################################################################################################################################################
    #visualize_reward_curve(rewards0_avg, rewards1_avg, filename_suffix0, foldern=paraset.foldername)
    if paraset.teamsize3:
        visualize_reward_curve(rewards0, rewards1, filename_suffix0 + "NO_AVG", foldern=paraset.foldername, bagsize=1, data2=rewards2)
    else:
        visualize_reward_curve(rewards0, rewards1, filename_suffix0 + "NO_AVG", foldern=paraset.foldername, bagsize=1)
    #visualize_curve(scores0_avg, scores1_avg, ylabel=f"Score (avg per {bagsize} episodes)", name=filename_suffix0, foldern=paraset.foldername)
    # the visualiztaion with bags is not necessary for the averaged score one? perhaps wrong, but I rather need some range and variance visualizations
    visualize_curve(scores0, scores1, ylabel=f"Score (NO AVG, scorelist_avg direct test)", name=filename_suffix0, foldern=paraset.foldername, bagsize=1) #bagsize 50 is standard, 1 has to be set here
    #visualize_curve(tags1_avg, tags0_avg, ylabel=f"Tags (avg per {bagsize} episodes)", name=filename_suffix0, foldern=paraset.foldername)
    visualize_curve(tags1, tags0, ylabel=f"Tags (NO AVG)", name=filename_suffix0, foldern=paraset.foldername, bagsize=1)
    matrix_to_heatmap(q_table, filename_suffix0, paraset.foldername + "figures/" + filename_suffix0 + "qheatmap.png", "Q-value (expected reward)") 
    # matrix_to_heatmap(s_table, filename_suffix0, "qtrainlog/"+FOLDER+"/figures/"+filename_suffix0+"visitcount.png", "States visited (count per state)")
    visualize_curve(winrate0, winrate1, ylabel=f"Winrate (avg per {bagsize} episodes)", name=filename_suffix0, foldern=paraset.foldername)
    ####################################################################################################################################################
    
    print(f"{filename_suffix0} Last avg. value - Rewards: {rewards0_avg[len(rewards0_avg) - 1]}, {rewards1_avg[len(rewards1_avg) - 1]}")
    print(f"{filename_suffix0} Last avg. value - Scores: {scores0_avg[len(scores0_avg) - 1]}, {scores1_avg[len(scores1_avg) - 1]}") # scores0_avg has length 20. 20 should not be the length here?!<-Yes it should be. It is just set to give episode500 value because of small batch5! Do I need to change the handling of the lists? thought it would be same list format but avg value instead of single value now...
    print(f"{filename_suffix0} Last avg. value - Winrate: {winrate0[len(winrate0) - 1]}, {winrate1[len(winrate1) - 1]}")
    print(f"{filename_suffix0} Last avg. value - Tags: {tags0_avg[len(tags0_avg) - 1]}, {tags1_avg[len(tags1_avg) - 1]}")



def single_training_visualizer(filename, max_index, TWENTY=20):
    """
    filename example:
    nothingslayer2_single_aggressive26_nothing_lrate0.1_discount0.9_initq10.0_1nrs_500ep_no_pre_nr0
    the files could then be called 
    nothingslayer2_single_aggressive26_nothing_lrate0.1_discount0.9_initq10.0_1nrs_500ep_no_pre_nr0_q_table_i499.npy
    thus max_index zB 499 is useful
    """
    # foldern = "qtrainlog/batch 3/"
    foldern = "qtrainlog/batch 4/"
    #filename = "qtrainlog/batch 3/" + filename
    # rewardcurve = np.load(f"{filename}_reward_curve_i{max_index}.npy")
    rewardcurve = np.load(f"{foldern}{filename}_reward_curve_i{max_index}.npy") #just for test
    scorelist = np.load(f"{foldern}{filename}_scores_i{max_index}.npy")

    print(f"rewardcurve shape: {np.shape(rewardcurve)}") #(500, 3)
    # print(f"scorelist shape: {np.shape(scorelist)}")


    # visualize_reward_quick_and_ugly(rewards0, rewards1, rewards2, filename_suffix0 + "NO_AVG", foldern=paraset.foldername, bagsize=1, data2=rewards2)
    # else:
    #     visualize_reward_quick_and_ugly(rewards0, rewards1, filename_suffix0 + "NO_AVG", foldern=paraset.foldername, bagsize=1)
    # #visualize_curve(scores0_avg, scores1_avg, ylabel=f"Score (avg per {bagsize} episodes)", name=filename_suffix0, foldern=paraset.foldername)
    # # the visualiztaion with bags is not necessary for the averaged score one? perhaps wrong, but I rather need some range and variance visualizations
    # visualize_curve(scores0, scores1, ylabel=f"Score (NO AVG, scorelist_avg direct test)", name=filename_suffix0, foldern=paraset.foldername, bagsize=1)



    # avg_rew_curve = np.mean(rewardcurve, axis=1) #average over the three agents, shape should be (500,) now
    # print(f"avg_rew_curve shape: {np.shape(avg_rew_curve)}")
    # print(f"avg_rew_curve: {avg_rew_curve}")

    #I have a rewardcurve of shape (500, 3) where the second dimension is the reward for each of the three agents. I want to separate the three agents into their own lists:
    rewards0 = rewardcurve[:, 0]
    rewards1 = rewardcurve[:, 1]
    rewards2 = rewardcurve[:, 2]
    
    visualize_reward_quick_and_ugly(rewards0, rewards1, rewards2, filename, bagsize=1)

    score0 = scorelist[:, 0]
    score1 = scorelist[:, 1]


    visualize_curve_quickandugly(score0, score1, ylabel="Score", name=filename, bagsize=1, foldern="qtrainlog/batch 3/quickfigures/")


    print("--------------------------------------------")
    print(f"{filename}\nLast avg. value - Rewards: {rewardcurve[len(rewardcurve) - 1]} (agents 0-2)")
    print(f"{filename}\nLast avg. value - Scores: {scorelist[len(scorelist) - 1]} (blue, red)")
    # print(f"{filename}\nLast avg. value - Winrate: {winrate0[len(winrate0) - 1]}, {winrate1[len(winrate1) - 1]}")

if __name__ == "__main__":
    
    #filename_suffix0 = "vshard_example_suffix01"
    #filename_suffix = "qtrainlog/vshard_"+filename_suffix
    
    #vis_helper(filename_suffix0)
    # print(f"Starting visualization of {FOLDER}.")
    #q_table = np.load("qtrainlog/vshard_example_suffix01_q_table.npy") #TODO visualize Q-Table
    #matrix_to_heatmap(q_table, "test_title", "qtrainlog/vshard_example_suffix01_q_table.png")

    # avg_vis_helper(ParameterSet("caps_and_tags", "hard", 0.1, 0.95, 0.0, False, "ratehigh", "qtrainlog/batch 7/", 0)) #testing the init q value 0 for capsntags
    # TODO add variance (see example qlearn paper)

    # avg_vis_helper(ParameterSet("aggr_rew_alt", "hard", 0.1, 0.9, 10.0, False, "testrew2", "qtrainlog/batch 7/", 0))
    # avg_vis_helper(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, False, "testrew2", "qtrainlog/batch 7/", 0))

    # avg_vis_helper(ParameterSet("single_aggressive_rew", "hard", 0.1, 0.9, 10.0, False, "math1test", "qtrainlog/batch 8/", 0, math2=False)) 
    # avg_vis_helper(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "math1test", "qtrainlog/batch 8/", 0, math2=False)) 
    # avg_vis_helper(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "math1test", "qtrainlog/batch 8/", 0, math2=False)) 

    i = 0 # do not forget to set other index if multiple indexes exist (not necessary for avg_vis_helper)
    filenames = []
    parametersets = []
    # batch 10
    #filenames.append(ParameterSet("aggressive_tags", "hard", 0.2, 0.95, 10.0, False, "testboolchange2", "qtrainlog/batch 10/", 0))
    # filenames.append(ParameterSet("aggressive_tags", "hard", 0.2, 0.95, 10.0, False, "testrestored", "qtrainlog/batch 10/", i))
    # filenames.append(ParameterSet("aggressive_tags", "hard", 0.1, 0.9, 10.0, False, "testboolchange2", "qtrainlog/batch 10/", i))
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "testboolchange2", "qtrainlog/batch 10/", i))
    # latest hpc-data (13.4.):
    # filenames.append(ParameterSet("aggressive_tags", "hard", 0.1, 0.9, 10.0, False, "testboolchange2", "qtrainlog/batch 10/", i))
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "testboolchange2", "qtrainlog/batch 10/", i))
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "testrestored2", "qtrainlog/batch 10/", i, boolchange=False)) #TODO compare boolchange with restored, and compare aggressive_tags with single_aggressive (not just score, also tags)
    # FROM HERE THE NEW NAME SCHEME IS USED IN TESTS
    # batch 10d (a lot of runs to gather solid statistics on chances of the training working) #NOTE currently not being run, not sure if I should proceed. Actually I must know what the success-rate of "normal" 0.2 0.95 oldbool single_aggr. is. Run this ~50 times.
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "x30boolchange", "qtrainlog/batch 10/", i))
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "x30restored", "qtrainlog/batch 10/", i, boolchange=False)) #TODO need better visualization before I commit to large test like this (serves no sufficient purpose rn)
    # batch 10d (fr tho, above lines were never run...)
    # filenames.append(ParameterSet("caps_and_tags", "hard", 0.1, 0.9, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=20, ep=1000)) #see if this improves "chasing" behaviour.
    # filenames.append(ParameterSet("aggressive_tags", "hard", 0.01, 0.95, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
    # filenames.append(ParameterSet("aggressive_tags", "hard", 0.2, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.01, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000))
    # filenames.append(ParameterSet("aggressive_tags", "hard", 0.01, 0.99, 10.0, False, "boolchange", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000)) 
    # batch 11
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "3v3test", "qtrainlog/batch 10/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True)) #(wrong folder, ups)

    # batch 1
    # filenames.append(ParameterSet("aggressive_tags_26", "hard", 0.1, 0.99, 10.0, False, "3v3test_newbool", "qtrainlog/batch 1/", i, boolchange=True, nrs=10, ep=600, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
    # filenames.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.95, 10.0, False, "3v3test_newbool", "qtrainlog/batch 1/", i, boolchange=True, nrs=10, ep=600, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.95, 10.0, False, "3v3test_oldbool", "qtrainlog/batch 1/", i, boolchange=False, nrs=10, ep=600, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
    # batch 2
    # filenames.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "3v3test_newbool", "qtrainlog/batch 2/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
    #         #                           mode changed from "hard" to "nothing" for now, because 26env doesnt seem to work with 24-base-policy. Training against ultra-defensive to verify training process...
    # filenames.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.95, 10.0, False, "3v3test_newbool", "qtrainlog/batch 2/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
    # filenames.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.95, 10.0, False, "3v3test_oldbool", "qtrainlog/batch 2/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True))
    # # batch 2b
    # filenames.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.95, 10.0, False, "speeduptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True)) #DO NOT forget to change the folder (or create it...)
    # filenames.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.95, 10.0, False, "speeduptest", "qtrainlog/batch 2/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True))
    # batch 2c
    # filenames.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True)) 
    # filenames.append(ParameterSet("aggressive_tags_26", "hard", 0.1, 0.99, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True)) 
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.2, 0.85, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
    # filenames.append(ParameterSet("single_aggressive_rew", "hard", 0.1, 0.9, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
    # filenames.append(ParameterSet("aggressive_tags_26", "hard", 0.1, 0.9, 10.0, False, "longeptest", "qtrainlog/batch 2/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True)) 
    # batch 3a1
    # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.95, 10.0, False, "stepstest", "qtrainlog/batch 3/", i, boolchange=False, nrs=3, ep=1000, teamsize3=True)) 
    # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.9, 10.0, False, "stepstest", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=1000, teamsize3=True)) 
    # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.99, 10.0, False, "stepstest", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=1000, teamsize3=True)) 
    # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "stepstest", "qtrainlog/batch 3/", i, boolchange=True, nrs=3, ep=1000, teamsize3=True)) 
    # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.95, 10.0, False, "stepstest", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=1000, teamsize3=True)) 
    # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.9, 10.0, False, "stepstest", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=1000, teamsize3=True)) 
    # # batch 3a2
    # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.2, 0.95, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=False, nrs=3, ep=600, teamsize3=True)) 
    # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.9, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
    # parametersets.append(ParameterSet("single_aggressive_rew", "nothing", 0.1, 0.99, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
    # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.1, 0.99, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=True, nrs=3, ep=600, teamsize3=True)) 
    # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.95, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=600, teamsize3=True)) 
    # parametersets.append(ParameterSet("aggressive_tags_26", "nothing", 0.01, 0.9, 10.0, False, "stepstest2", "qtrainlog/batch 3/", i, boolchange=False, nrs=1, ep=600, teamsize3=True)) 

    # # batch 3c
    # nbrs = 20
    # filenames.append(ParameterSet("aggressive_tags_26", "hard", 0.2, 0.95, 10.0, False, "single26test", "qtrainlog/batch 3/", 0, boolchange=False, nrs=nbrs, ep=700, teamsize3=True, timelimit=600.)) 
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "single26test", "qtrainlog/batch 3/", 0, boolchange=False, nrs=nbrs, ep=700, teamsize3=True, timelimit=600.)) 
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.99, 10.0, False, "single26test", "qtrainlog/batch 3/", 0, boolchange=True, nrs=nbrs, ep=700, teamsize3=True, timelimit=600.)) 

    # #batch 3d
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrab", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=300, teamsize3=True)) 
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=300, teamsize3=True)) 

    # #nothingslayer
    # filenames.append(ParameterSet("single_aggressive26", "nothing", 0.1, 0.9, 10.0, False, "nothingslayer1", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=400, teamsize3=True)) 
    # filenames.append(ParameterSet("single_aggressive26", "nothing", 0.1, 0.9, 10.0, False, "nothingslayer2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=1, ep=500, teamsize3=True, ignoreseed=False, timelimit=3000.))

    # Zeitnot-comp training
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong3", "qtrainlog/batch 3/", 0, boolchange=False, nrs=20, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong4", "qtrainlog/batch 3/", 0, boolchange=True, nrs=10, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=5, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
    # filenames.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong3", "qtrainlog/batch 3/", 0, boolchange=True, nrs=5, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 

    # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "fromzero3000sec", "qtrainlog/batch 4/", 0, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=3000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
    # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong4", "qtrainlog/batch 4/", 0, boolchange=False, nrs=20, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
    # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong5", "qtrainlog/batch 4/", 0, boolchange=True, nrs=20, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
    # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong2", "qtrainlog/batch 3/", 0, boolchange=False, nrs=20, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 
    # parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.9, 10.0, False, "enhancegrablong3", "qtrainlog/batch 3/", 0, boolchange=True, nrs=20, ep=300, teamsize3=True, timelimit=6000., qtable_suffix="qtrainlog/batch 3/enhancegrab_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_1nrs_300ep_no_pre_nr0_q_table.npy")) 

    # #scattershot hpc 23-04-26 eod ending (ca)
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch1", "qtrainlog/batach 4/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch2", "qtrainlog/batach 4/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parametersearch3", "qtrainlog/batach 4/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parametersearch4", "qtrainlog/batach 4/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "parametersearch5", "qtrainlog/batach 4/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "parametersearch6", "qtrainlog/batach 4/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch7_newbool", "qtrainlog/batach 4/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))
    parametersets.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch8_newbool", "qtrainlog/batach 4/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False))


    for e in filenames:
        # check if folder is already created
        if not os.path.isdir(e.foldername+"figures/"):
            print("Creating folder "+e.foldername+"figures/")
            os.makedirs(e.foldername+"figures/")
        #vis_helper(e)
        avg_vis_helper(e)
    for e in parametersets:
        # check if folder is already created
        if not os.path.isdir(e.foldername+"figures/"):
            print("Creating folder "+e.foldername+"figures/")
            os.makedirs(e.foldername+"figures/")
        avg_vis_helper(e)
    #vis_helper("math2scatter_aggressive_tags_24_hard_lrate0.1_discount0.9_initq10.0_1000ep_no_pre_nr0")

    # """Finding out how dimensions are mapped onto qtable matrix heatmap:
    # (toggle comment below to generate markers for the dimension [currently dim. f])"""
    # qtable = np.zeros((4, 4, 4, 2, 2, 4))
    # for a in range(4):
    #     for b in range(4):
    #         for c in range(4):
    #             for d in range(2):
    #                 for e in range(2):
    #                     for f in range(4):
    #                         if f == 0:
    #                             qtable[a][b][c][d][e][f] = 10
    # na = "f"
    # matrix_to_heatmap(qtable, f"dimension {na}", f"dimension_{na}_legend", "name")

    #single_training_visualizer("nothingslayer2_single_aggressive26_nothing_lrate0.1_discount0.9_initq10.0_1nrs_500ep_no_pre_nr0", 499, TWENTY=1)
    # single_training_visualizer("enhancegrablong3_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_5nrs_300ep_no_pre_nr0", 9)
    
    
    
    
    
    
    
    # single_training_visualizer("enhancegrablong2_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_5nrs_300ep_no_pre_nr0", 24)
    # single_training_visualizer("enhancegrablong2_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_5nrs_300ep_no_pre_nr0", 299)
    # single_training_visualizer("enhancegrablong3_single_aggressive26_hard_lrate0.1_discount0.9_initq10.0_5nrs_300ep_no_pre_nr0", 299)

    # single_training_visualizer("fromzero3000sec_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_10nrs_1000ep_no_pre_nr9", 999, TWENTY=1)
    # single_training_visualizer("fromzero3000sec_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_10nrs_1000ep_no_pre_nr8", 999, TWENTY=1)
    # single_training_visualizer("fromzero3000sec_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_10nrs_1000ep_no_pre_nr6", 999, TWENTY=1)
    # single_training_visualizer("fromzero3000sec_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_10nrs_1000ep_no_pre_nr5", 999, TWENTY=1)
    

    print("Ended visualization.")
