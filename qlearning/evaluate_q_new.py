import os
from matplotlib import pyplot as plt
import numpy as np
from train_qlearn import ParameterSet

def plot_rewards(rewarray, foldername): #TODO add parameters that are important for labeling/naming TODO also add parameter for location where plot should be saved
    """Rewardcurve visualization, rewarray is a list, that contains lists of rewards (per-episode) where each entry consists of 3 entries for the 3 agents of a game.
    Current method is to sum up the 3 agents rewards, removing the last dimension from the above.
    number should correlate with the length of rewarray (dimension 0), so it is removed from the required input."""
    print("rewarrray shape in plot_rewards:", np.shape(rewarray))
    # print("rewarrray:", rewarray)

    #sum up the agents:
    sum_array = np.sum(np.array(rewarray), 2) #is 2 correct? yes.
    # print("rewarrray(agents are summed):", mrewarray, "shape:", np.shape(mrewarray))

    meandata = np.mean(np.array(sum_array), 0) #is 0 correct? yes, and dim flattened correctly.
    # print("rewarrray(agents are averaged):", avrewarray, "shape:", np.shape(avrewarray))

    print("meandata:",meandata)
    lowdata = meandata - np.std(sum_array, 0)
    print("lowdata:", lowdata)









def visualize_one_reward_curve(meandata, lowdata, highdata, foldername):
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


if __name__ == "__main__":
    print("~ visualization 2.0 - work in progress ~")

    #loading in realistic data from newest batch
    folder = "qtrainlog/batch 6b/"
    reward_name = ["shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_reward_curve.npy", 
                "shortpara2_newbool_aggressive_tags_26_hard_lrate0.01_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_reward_curve.npy"]
    scores_name = ["shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_scores.npy", 
                "shortpara2_newbool_aggressive_tags_26_hard_lrate0.01_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_scores.npy"]
    # load data from file
    rewardcurve = np.array([np.load(folder + reward_name[i]) for i in range(2)])
    scorelist = np.array([np.load(folder + scores_name[i]) for i in range(2)])

    #shortening dimensions for readable test prints:    (should be removed after testing)
    print(f"Shape of rewardcurve: {np.shape(rewardcurve)}")
    # cut dimension 1 to length 4

    rewardcurve = rewardcurve[:, :5, :]#.copy() #copy not necessary?

    print(f"Shape of rewardcurve post-shortening: {np.shape(rewardcurve)}")


    # testing 
    plot_rewards(rewardcurve, folder)
    
    # read in the provided filename and process it with a function (so we can call this function from elsewhere if we choose to automate this)
    

    # code to use to make the figures folder create itself if forgotten...
    # if not os.path.isdir(e.foldername+"figures/"):
    #     print("Creating folder "+e.foldername+"figures/")
    #     os.makedirs(e.foldername+"figures/")