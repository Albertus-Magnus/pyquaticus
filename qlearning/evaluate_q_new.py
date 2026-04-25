import os
from matplotlib import pyplot as plt
import numpy as np
from train_qlearn import ParameterSet

def plot_rewards(rewarray, foldername, name):
    """Rewardcurve visualization, rewarray is a list, that contains lists of rewards (per-episode) where each entry consists of 3 entries for the 3 agents of a game.
    Current method is to sum up the 3 agents rewards, removing the last dimension from the above and using a team reward thus.
    """
    sum_array = np.sum(np.array(rewarray), 2)
    meandata = np.mean(np.array(sum_array), 0) 
    lowdata = meandata - np.std(sum_array, 0)
    highdata = meandata + np.std(sum_array, 0)
    visualize_reward_curve(meandata, lowdata, highdata, foldername, name)
#End of plot_rewards()


def plot_anythingelse(scorearray, foldername, name, attribute_name):
    """General visualization, scorearray is a list, that contains lists of scores (per-episode) where each entry consists of 2 entries for the 2 teams of a game.
    """
    meandata = np.mean(np.array(scorearray), 0)
    stddata = np.std(np.array(scorearray), 0) 
    lowdata = meandata - stddata
    highdata = meandata + stddata
    visualize_curve(meandata, lowdata, highdata, foldername, name, attribute_name)
    visualize_many_curves(scorearray, foldername, name, attribute_name)
#End of plot_rewards()


def visualize_curve(meandata, lowdata, highdata, foldername, name, attribute_name):
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    # print("meandata shape:", np.shape(meandata), "lowdata shape:", np.shape(lowdata), "highdata shape:", np.shape(highdata))
    plt.figure(figsize=(12, 6))
    plt.plot([meandata[i][0] for i in range(len(meandata))], label='Team Blue', color='blue')
    plt.fill_between(range(len(lowdata)), [lowdata[i][0] for i in range(len(lowdata))], [highdata[i][0] for i in range(len(highdata))], alpha=0.4, color='blue', label='Standard Deviation between training attempts')
    plt.plot([meandata[i][1] for i in range(len(meandata))], label='Team Red', color='red')
    plt.fill_between(range(len(lowdata)), [lowdata[i][1] for i in range(len(lowdata))], [highdata[i][1] for i in range(len(highdata))], alpha=0.4, color='red', label='Standard Deviation between training attempts')
    plt.xlabel("Episodes")
    plt.ylabel(attribute_name)
    plt.title(f"{attribute_name}\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    plt.gca().set_xticks(ticks)
    # plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldername}figures/{name}_{attribute_name}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
#End of visualize_curve()


# def visualize_many_curves(data: np.ndarray, name, ylabel="Score", foldern=f"qtrainlog/folder/"):
def visualize_many_curves(data, foldername, name, attribute_name):
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    # print("meandata shape:", np.shape(data), "lowdata shape:", np.shape(lowdata), "highdata shape:", np.shape(highdata))
    plt.figure(figsize=(12, 6))
    for p in range(len(data)):
        plt.plot([data[p][i][0] for i in range(len(data[p]))], color='blue', alpha=0.6)
    # plt.fill_between(range(len(lowdata)), [lowdata[i][0] for i in range(len(lowdata))], [highdata[i][0] for i in range(len(highdata))], alpha=0.2, color='blue', label='Standard Deviation between training attempts')
    plt.plot([data[p][i][1] for i in range(len(data[p]))], color='red', alpha=0.6)
    # plt.fill_between(range(len(lowdata)), [lowdata[i][1] for i in range(len(lowdata))], [highdata[i][1] for i in range(len(highdata))], alpha=0.2, color='red', label='Standard Deviation between training attempts')
    plt.xlabel("Episodes")
    plt.ylabel(attribute_name)
    plt.title(f"{attribute_name}\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    plt.gca().set_xticks(ticks)
    # plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    # plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldername}figures/{name}_{attribute_name}_manyfold.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close() #
#End of visualize_many_curves()


def visualize_reward_curve(meandata, lowdata, highdata, foldername, name):
    # reward_curve = np.load(reward_curve_file, allow_pickle=True)
    # rewards0 = [step[0] for step in reward_curve]
    # rewards1 = [step[1] for step in reward_curve]
    plt.figure(figsize=(12, 6))
    plt.plot(meandata, label='Mean of team rewards', color='blue')
    plt.fill_between(range(len(lowdata)), lowdata, highdata, alpha=0.2, color='blue', label='Standard Deviation between training attempts')
    # plt.plot(data1, label='Agent 1', color='darkblue')
    # if data2 is not None:
    #     plt.plot(data2, label='Agent 2', color='lightblue')
    plt.xlabel(f"Episodes\n(games between qlearning updates)")
    plt.ylabel(f"Reward\n(sum of agents per team)")
    plt.title(f"Rewards during Q-Learn Training\n{name}")
    plt.grid(True)
    # Scale x-achsis labels times 50 (or times bagsize):
    ticks = plt.gca().get_xticks()
    ticks = ticks[1:(len(ticks)-1)]
    #print("\nticks: ",ticks)
    plt.gca().set_xticks(ticks)
    # plt.gca().set_xticklabels([f'{int(x*bagsize)}' for x in ticks])
    plt.legend()
    # Save figure to file:
    plt.savefig(f"{foldername}figures/{name}_reward.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close() #
#End of visualize_reward_curve()


def load_and_call_helper(name, nrs, folder):
    name_indexed = []
    for i in range(nrs):
        name_indexed.append(name + f"_nr{i}")
    reward_name = [f"{nam}_reward_curve.npy" for nam in name_indexed]
    scores_name = [f"{nam}_scores.npy" for nam in name_indexed]

    # test names, now redundant
    # reward_name = ["shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_reward_curve.npy", 
    #             "shortpara2_newbool_aggressive_tags_26_hard_lrate0.01_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_reward_curve.npy"]
    # scores_name = ["shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_scores.npy", 
    #             "shortpara2_newbool_aggressive_tags_26_hard_lrate0.01_discount0.99_initq10.0_1nrs_600ep_no_pre_nr0_scores.npy"]

    # load data from file
    rewardcurve = np.array([np.load(folder + reward_name[i]) for i in range(len(reward_name))])
    scorelist = np.array([np.load(folder + scores_name[i]) for i in range(len(scores_name))])

    print(rewardcurve) #TODO issue: the arrays of different parallel trainings are the same?!? THIS IS BAD!

    #shortening dimensions for readable test prints:    (should be removed after testing)
    # rewardcurve = rewardcurve[:, 5:10, :]#.copy() #copy not necessary?
    # scorelist = scorelist[:, 5:10, :]

    if not os.path.isdir(folder+"figures/"):
        print("Creating folder "+folder+"figures/")
        os.makedirs(folder+"figures/")

    plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])
    plot_anythingelse(scorelist, folder, scores_name[0][:-(len("_nr0_scores.npy"))], "Score")


if __name__ == "__main__":
    v1 = np.load("qtrainlog/batch 6c/parameterconfirm18_newbool_aggressive_tags_26_hard_lrate0.15_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_scores.npy")
    for i in range(len(v1)):
        print(i,": ", v1[i])
    # TODO CONCLUSION: All 5 parallel training experiments resulted in the exact same results (qtable, scores, reward, ...), thus rendering 4 out of 5 calculations trash.
    # WHY is that so? It looks like ignoreseed is set correctly, and it definitely works in principal in the new 26env. 
    
    # Also, I don't have time to re-run the calculations before the competition deadline, but I can (TODO) rerun it before the meeting (nrs 20)...

    import sys
    sys.exit()
    print("~ visualization 2.0 ~")

    #loading in realistic data from newest batch
    # folder = "qtrainlog/batch 6b/"

    names: list[ParameterSet] = []
    i = 0
    ######################################################################
    # # batch 6b
    # #load_and_call_helper("shortpara1_sharpturns_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_1nrs_600ep_no_pre", 1)#_nr0") #alternative way to call the visualization
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara1", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=True, sim_speedup=3))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "shortpara2", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "shortpara3", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # # comparison with sharpturns (here non-sharp)
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara4", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "shortpara5", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "shortpara6", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "shortpara7", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "shortpara8", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "shortpara9", "qtrainlog/batch 6b/", 0, boolchange=False, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "shortpara10", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "shortpara11", "qtrainlog/batch 6b/", 0, boolchange=True, nrs=1, ep=600, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # # batch 6a
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parametersearch17", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parametersearch18", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch9", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch10", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parametersearch11", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parametersearch12", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "parametersearch13", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.001, 0.85, 10.0, False, "parametersearch14", "qtrainlog/batch 6/", i, boolchange=False, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parametersearch15", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parametersearch16", "qtrainlog/batch 6/", i, boolchange=True, nrs=10, ep=1000, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3))

    # batch 6c
    names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # batch 6d
    names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    




    #####################################################################
    for name in names:
        load_and_call_helper(name.create_name_without_index(), name.nrs, name.foldername)

    



    # testing 
    # plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])
    # plot_anythingelse(scorelist, folder, scores_name[0][:-(len("_nr0_scores.npy"))], "Score")
    # for rew in reward_name:
    #     plot_rewards(np.array([np.load(folder + rew) for rew in reward_name]), folder, rew[:-(len("_nr0_reward_curve.npy"))])
    #     plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])