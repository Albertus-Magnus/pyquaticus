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
    print("plotting anythingelse")
    meandata = np.mean(np.array(scorearray), 0)
    stddata = np.std(np.array(scorearray), 0) 
    lowdata = meandata - stddata
    highdata = meandata + stddata
    visualize_curve(meandata, lowdata, highdata, foldername, name, attribute_name)
    #visualize_many_curves(scorearray, foldername, name, attribute_name)
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
    plt.savefig(f"{foldername}figures/{name}_{attribute_name}.png", dpi=300, bbox_inches='tight') #NOTE changed output folder for different structure: figures and 
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

    # print("rewardcurve:", rewardcurve) #debug print

    #shortening dimensions for readable test prints:    (should be removed after testing)
    # rewardcurve = rewardcurve[:, 5:10, :]#.copy() #copy not necessary?
    # scorelist = scorelist[:, 5:10, :]

    if not os.path.isdir(folder+"figures/"):
        print("Creating folder "+folder+"figures/")
        os.makedirs(folder+"figures/")

    plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_reward_curve.npy"))])
    plot_anythingelse(scorelist, folder, scores_name[0][:-(len("_scores.npy"))], "Score")

    # Compute for every i (0 to 19) the average team score for the last 100 episodes and print it, to get a quick overview of the final performance of the training.
    for i in range(len(scorelist)):
        final_scores = scorelist[i][-100:] # last 100 episodes
        avg_team_score = np.mean(final_scores, axis=0) # average over episodes, resulting in average score for team blue and team red
        print(f"Final average team scores for {name_indexed[i]}: Team Blue: {avg_team_score[0]:.2f}, Team Red: {avg_team_score[1]:.2f}")
    print(f"\nHighest final average team score for {name}: {max(np.mean(scorelist[i][-100:], axis=0)[0] for i in range(len(scorelist))):.2f} (Team Blue), smallest opponent score was {min(np.mean(scorelist[i][-100:], axis=0)[1] for i in range(len(scorelist))):.2f} (Team Red Minimum)")

    print(scores_name[0])


# lines 145-333 ergo ~150 positions that result in correct circle detection
example_positions = np.array([[50.03873777, 59.99915642] ,
    [50.15176609, 59.98705141] ,
    [50.33354455, 59.93724541] ,
    [50.57023648, 59.82176   ] ,
    [50.84125533, 59.61936747] ,
    [51.12028534, 59.31469225] ,
    [51.3767984 , 58.89951357] ,
    [51.57786684, 58.37373028] ,
    [51.69017451, 57.74588901] ,
    [51.68212486, 57.03323852] ,
    [51.52593873, 56.26129551] ,
    [51.19963256, 55.46293107] ,
    [50.68877198, 54.67701052] ,
    [49.98790402, 53.94664167] ,
    [49.10158456, 53.31710689] ,
    [48.04493509, 52.83357141] ,
    [46.84368365, 52.53867335] ,
    [45.53366805, 52.47010959] ,
    [44.15980424, 52.658335  ] ,
    [42.77454796, 53.12449072] ,
    [41.43590243, 53.87867009] ,
    [40.20892068, 54.91485728] ,
    [39.18800247, 56.17345709] ,
    [38.42100561, 57.60106464] ,
    [37.93510136, 59.14710623] ,
    [37.74750311, 60.75681255] ,
    [37.86485662, 62.37315894] ,
    [38.2830046 , 63.93888555] ,
    [38.98713393, 65.39852571] ,
    [39.95230049, 66.70037096] ,
    [41.14431277, 67.79830276] ,
    [42.52094314, 68.65342635] ,
    [44.03342378, 69.23544854] ,
    [45.62817432, 69.52375091] ,
    [47.24869992, 69.50812019] ,
    [48.83759265, 69.18911012] ,
    [50.33856519, 68.57802179] ,
    [51.69844485, 67.69650331] ,
    [52.86905721, 66.5757829 ] ,
    [53.80893274, 65.25556265] ,
    [54.48477589, 63.78261202] ,
    [54.87264457, 62.20911101] ,
    [54.95879834, 60.59080169] ,
    [54.74018517, 58.98501347] ,
    [54.22454953, 57.4486322 ] ,
    [53.43015807, 56.03608495] ,
    [52.3851525 , 54.79741191] ,
    [51.12655268, 53.77649369] ,
    [49.69894512, 53.00949683] ,
    [48.15290354, 52.52359257] ,
    [46.54319722, 52.33599432] ,
    [44.92685082, 52.45334783] ,
    [43.36112422, 52.87149581] ,
    [41.90148405, 53.57562514] ,
    [40.59963881, 54.5407917 ] ,
    [39.501707  , 55.73280398] ,
    [38.64658341, 57.10943435] ,
    [38.06456122, 58.621915  ] ,
    [37.77625886, 60.21666553] ,
    [37.79188957, 61.83719113] ,
    [38.11089964, 63.42608386] ,
    [38.72198797, 64.9270564 ] ,
    [39.60350646, 66.28693606] ,
    [40.72422687, 67.45754842] ,
    [42.04444712, 68.39742396] ,
    [43.51739775, 69.0732671 ] ,
    [45.09089876, 69.46113578] ,
    [46.70920808, 69.54728956] ,
    [48.3149963 , 69.32867638] ,
    [49.85137757, 68.81304074] ,
    [51.26392482, 68.01864928] ,
    [52.50259786, 66.97364372] ,
    [53.52351608, 65.71504389] ,
    [54.29051294, 64.28743633] ,
    [54.77641719, 62.74139475] ,
    [54.96401545, 61.13168843] ,
    [54.84666193, 59.51534203] ,
    [54.42851396, 57.94961543] ,
    [53.72438462, 56.48997526] ,
    [52.75921806, 55.18813002] ,
    [51.56720578, 54.09019821] ,
    [50.19057541, 53.23507463] ,
    [48.67809477, 52.65305244] ,
    [47.08334423, 52.36475007] ,
    [45.46281863, 52.38038079] ,
    [43.87392591, 52.69939085] ,
    [42.37295336, 53.31047919] ,
    [41.0130737 , 54.19199767] ,
    [39.84246134, 55.31271808] ,
    [38.90258581, 56.63293833] ,
    [38.22674266, 58.10588896] ,
    [37.83887398, 59.67938997] ,
    [37.75272021, 61.29769929] ,
    [37.97133338, 62.90348751] ,
    [38.48696902, 64.43986878] ,
    [39.28136048, 65.85241603] ,
    [40.32636605, 67.09108907] ,
    [41.58496587, 68.11200729] ,
    [43.01257343, 68.87900415] ,
    [44.55861502, 69.36490841] ,
    [46.16832133, 69.55250666] ,
    [47.78466773, 69.43515314] ,
    [49.35039434, 69.01700517] ,
    [50.8100345, 68.31287584] ,
    [52.11187974, 67.34770928] ,
    [53.20981155, 66.155697] ,
    [54.06493514, 64.77906662] ,
    [54.64695733, 63.26658598] ,
    [54.93525969, 61.67183544] ,
    [54.91962898, 60.05130985] ,
    [54.60061891, 58.46241712] ,
    [53.98953058, 56.96144457] ,
    [53.10801209, 55.60156491] ,
    [51.98729169, 54.43095255] ,
    [50.66707144, 53.49107702] ,
    [49.1941208, 52.81523387] ,
    [47.6206198, 52.42736519] ,
    [46.00231048, 52.34121142] ,
    [44.39652226, 52.55982459] ,
    [42.86014098, 53.07546024] ,
    [41.44759373, 53.8698517] ,
    [40.20892069, 54.91485726] ,
    [39.18800247, 56.17345709] ,
    [38.42100561, 57.60106464] ,
    [37.93510136, 59.14710623] ,
    [37.74750311, 60.75681255] ,
    [37.86485662, 62.37315894] ,
    [38.2830046, 63.93888555] ,
    [38.98713393, 65.39852571] ,
    [39.95230049, 66.70037096] ,
    [41.14431277, 67.79830276] ,
    [42.52094314, 68.65342635] ,
    [44.03342378, 69.23544854] ,
    [45.62817432, 69.52375091] ,
    [47.24869992, 69.50812019] ,
    [48.83759265, 69.18911012] ,
    [50.33856519, 68.57802179] ,
    [51.69844485, 67.69650331] ,
    [52.86905721, 66.5757829 ] ,
    [53.80893274, 65.25556265] ,
    [54.48477589, 63.78261202] ,
    [54.87264457, 62.20911101] ,
    [54.95879834, 60.59080169] ,
    [54.74018517, 58.98501347] ,
    [54.22454953, 57.4486322 ] ,
    [53.43015807, 56.03608495] ,
    [52.3851525, 54.79741191] ,
    [51.12655268, 53.77649369] ,
    [49.69894512, 53.00949683] ,
    [48.15290354, 52.52359257] ,
    [46.54319722, 52.33599432] ,
    [44.92685082, 52.45334783] ,
    [43.36112422, 52.87149581] ,
    [41.90148405, 53.57562514] ,
    [40.59963881, 54.5407917 ] ,
    [39.501707, 55.73280398] ,
    [38.64658341, 57.10943435] ,
    [38.06456122, 58.621915  ] ,
    [37.77625886, 60.21666553] ,
    [37.79188957, 61.83719113] ,
    [38.11089964, 63.42608386] ,
    [38.72198797, 64.9270564 ] ,
    [39.60350646, 66.28693606] ,
    [40.72422687, 67.45754842] ,
    [42.04444712, 68.39742396] ,
    [43.51739775, 69.0732671 ] ,
    [45.09089876, 69.46113578] ,
    [46.70920808, 69.54728956] ,
    [48.3149963, 69.32867638] ,
    [49.85137757, 68.81304074] ,
    [51.26392482, 68.01864928] ,
    [52.50259786, 66.97364372] ,
    [53.52351608, 65.71504389] ,
    [54.29051294, 64.28743633] ,
    [54.77641719, 62.74139475] ,
    [54.96401545, 61.13168843] ,
    [54.84666193, 59.51534203] ,
    [54.42851396, 57.94961543] ,
    [53.72438462, 56.48997526] ,
    [52.75921806, 55.18813002] ,
    [51.56720578, 54.09019821] ,
    [50.19057541, 53.23507463] ,
    [48.67809477, 52.65305244] ,
    [47.08334423, 52.36475007] ,
    [45.46281863, 52.38038079] ,
    [43.87392591, 52.69939085] ,
    [42.2837744, 53.15842914] ,
    [40.6220698, 53.63778441] ,
    [38.99913018, 54.10595707] ])

#lines 336-414 ergo ~80 position that result in correctly detecting no circle.
example2 = np.array([[ 50.0 , 60.0 ],
    [ 50.03874999999999 , 60.0 ],
    [ 50.152499999999996 , 60.0 ],
    [ 50.341249999999995 , 60.0 ],
    [ 50.605 , 60.0 ],
    [ 50.94145620086686 , 59.965223068944645 ],
    [ 51.337233753868205 , 59.84671203832257 ],
    [ 51.7703380284345 , 59.62178323231327 ],
    [ 52.2125508897161 , 59.27346674299434 ],
    [ 52.630886643554135 , 58.79201802726078 ],
    [ 53.04998083253635 , 58.222521422517985 ],
    [ 53.5136442706373 , 57.592703546901404 ],
    [ 54.02177197025033 , 56.9024875798565 ],
    [ 54.574363931375444 , 56.15187352138326 ],
    [ 55.17142015401262 , 55.34086137148169 ],
    [ 55.81294063816189 , 54.46945113015178 ],
    [ 56.583367776726575 , 53.617411819182834 ],
    [ 57.559575919007386 , 52.87969300771314 ],
    [ 58.72384523379509 , 52.304744116239256 ],
    [ 60.047380617159554 , 51.93808623227965 ],
    [ 61.46560820094567 , 51.66782272201634 ],
    [ 62.95745696298982 , 51.38323735048927 ],
    [ 64.52297726883863 , 51.08459838036828 ],
    [ 66.07700109033145 , 50.62424617391964 ],
    [ 67.51700855097499 , 49.88079070008184 ],
    [ 68.79219985649547 , 48.88067337565798 ],
    [ 69.85740071915777 , 47.65932385546679 ],
    [ 70.6748758673541 , 46.26000905515588 ],
    [ 71.21566583641338 , 44.732300399144314 ],
    [ 71.46061287207483 , 43.13031772819919 ],
    [ 71.4010396034836 , 41.510812077095395 ],
    [ 71.03905644337935 , 39.93115524091585 ],
    [ 70.38748682567281 , 38.447307350608504 ],
    [ 69.59899730211534 , 36.99211588992188 ],
    [ 68.77506552325276 , 35.47082539693308 ],
    [ 68.1164626467598 , 33.90442967720396 ],
    [ 67.78102218806357 , 32.317707736091926 ],
    [ 67.74896814578415 , 30.69742378978312 ],
    [ 68.02109280458008 , 29.099833198184943 ],
    [ 68.58775601274769 , 27.581531404470326 ],
    [ 69.42888344363188 , 26.196305006619784 ],
    [ 70.51467773865564 , 24.993226340371564 ],
    [ 71.80667409360309 , 24.01491506726351 ],
    [ 73.25910289348494 , 23.296028351875332 ],
    [ 74.82051112396772 , 22.86203311449534 ],
    [ 76.435585120058 , 22.728303852772832 ],
    [ 78.04711008026105 , 22.899577992474057 ],
    [ 79.59799692944782 , 23.369788061812642 ],
    [ 81.13673808010239 , 23.979356423276233 ],
    [ 82.74532888206288 , 24.616242576466774 ],
    [ 84.423652920465 , 25.280738071108665 ],
    [ 86.17171019530873 , 25.97284290720191 ],
    [ 87.98950070659407 , 26.6925570847465 ],
    [ 89.83318743004578 , 27.422524298009517 ],
    [ 91.5697580912201 , 28.110081262024803 ],
    [ 93.19010669165837 , 28.7516226569545 ],
    [ 94.69423323136061 , 29.34714848279861 ],
    [ 96.08213771032685 , 29.896658739557136 ],
    [ 97.35382012855705 , 30.40015342723008 ],
    [ 98.5092804860512 , 30.857632545817438 ],
    [ 99.54851878280932 , 31.269096095319213 ],
    [ 100.4715350188314 , 31.634544075735405 ],
    [ 101.27832919411746 , 31.953976487066015 ],
    [ 101.9689013086675 , 32.227393329311035 ],
    [ 102.5432513624815 , 32.45479460247048 ],
    [ 103.00137935555946 , 32.63618030654433 ],
    [ 103.3432852879014 , 32.771550441532604 ],
    [ 103.56807427523215 , 32.86310410930642 ],
    [ 103.7338964842236 , 32.94484021915958 ],
    [ 103.94389137087295 , 33.0868287212271 ],
    [ 104.16890964179645 , 33.31634818893915 ],
    [ 104.35815642110096 , 33.65222901470473 ],
    [ 104.45265856590234 , 34.097981002558434 ],
    [ 104.41308190565583 , 34.627361166771124 ],
    [ 104.22218333860835 , 35.20259608459778 ],
    [ 103.87589910755801 , 35.78932182477819 ],
    [ 103.37855143161454 , 36.35932692330281 ],
    [ 102.73912208445485 , 36.8910754594871 ],
    [ 101.96859219537836 , 37.36912493814795 ]])

def circle_detection(positions: np.ndarray) -> bool:
    # Run a check if the trace of positions in example_positions is just going in circles the entire time. If not this should return False.
    
    # Calculate distances between consecutive points
    # distances = np.linalg.norm(np.diff(example_positions, axis=0), axis=1)
    # mean_dist = np.mean(distances)
    # std_dist = np.std(distances)

    # Calculate distances from centroid
    centroid = np.mean(positions, axis=0)
    distances_from_center = np.linalg.norm(positions - centroid, axis=1)
    mean_radius = np.mean(distances_from_center)
    std_radius = np.std(distances_from_center)

    # Check if:
    # 1. Distances between consecutive points are relatively consistent
    # 2. All points are roughly equidistant from centroid (low std relative to mean)
    # 3. The path closes (start and end points are similar)
    #is_circle = (std_dist / mean_dist < 0.2 and 
    is_circle = std_radius / mean_radius < 0.15 #and
                #  np.linalg.norm(example_positions[0] - example_positions[-1]) < mean_dist * 2)

    return is_circle

if __name__ == "__main__":

    # print("Circledetector test")
    # print(circle_detection(example_positions))
    # print(circle_detection(example2))

    # v1 = np.load("qtrainlog/batch 6c/parameterconfirm18_newbool_aggressive_tags_26_hard_lrate0.15_discount0.95_initq10.0_5nrs_1000ep_no_pre_nr0_scores.npy")
    # # for i in range(len(v1)):
    #     # print(i,": ", v1[i])
    # # DONE (hardcoded to be random starts) CONCLUSION: All 5 parallel training experiments resulted in the exact same results (qtable, scores, reward, ...), thus rendering 4 out of 5 calculations trash.
    # # WHY is that so? It looks like ignoreseed is set correctly, and it definitely works in principal in the new 26env. 
    
    # # Also, I don't have time to re-run the calculations before the competition deadline, but I can (DONE) rerun it before the meeting (nrs 20)...

    # import sys
    # sys.exit()
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

    # # batch 6c
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=5, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # # batch 6d
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6c/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6c/", i, boolchange=True, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6c/", i, boolchange=False, nrs=2, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))

    # # batch 6e    
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm17", "qtrainlog/batch 6e/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm18", "qtrainlog/batch 6e/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "parameterconfirm9", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.9, 10.0, False, "parameterconfirm10", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "parameterconfirm11", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.15, 0.95, 10.0, False, "parameterconfirm12", "qtrainlog/batch 6e/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3))

    # batch 7a
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "previoustest0", "qtrainlog/batch 7a/", 0, boolchange=False, nrs=1, ep=10, teamsize3=True, timelimit=60., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "previoustest1", "qtrainlog/batch 7a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.99, 10.0, False, "previoustest2", "qtrainlog/batch 7a/", 0, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1500., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    # testing
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "deleteme", "qtrainlog/batch 7a/", i, boolchange=False, nrs=3, ep=3, teamsize3=True, timelimit=600., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "previoustest1", "qtrainlog/batch 7a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    # names.append(ParameterSet("single_aggressive26", "hard", 0.01, 0.95, 10.0, False, "previoustest2", "qtrainlog/batch 7a/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.01, 0.99, 10.0, False, "previoustest3", "qtrainlog/batch 7a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    # names.append(ParameterSet("aggressive_tags_26", "hard", 0.15, 0.95, 10.0, False, "previoustest4", "qtrainlog/batch 7a/", i, boolchange=True, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True))
    names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue1", "qtrainlog/batch 7b/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr10_q_table.npy"))
    names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue2", "qtrainlog/batch 7b/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 7a/previoustest1__prevact_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr8_q_table.npy"))
    names.append(ParameterSet("single_aggressive26", "hard", 0.1, 0.99, 10.0, False, "prevcontinue3", "qtrainlog/batch 7b/", i, boolchange=False, nrs=20, ep=1000, teamsize3=True, timelimit=1200., ignoreseed=False, sharpturns=False, sim_speedup=3, previous_action=True, qtable_suffix="qtrainlog/batch 6e/parameterconfirm9_single_aggressive26_hard_lrate0.1_discount0.99_initq10.0_20nrs_1000ep_no_pre_nr0_q_table.npy"))


    #####################################################################
    for name in names:
        load_and_call_helper(name.create_name_without_index(), name.nrs, name.foldername)

    



    # testing 
    # plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])
    # plot_anythingelse(scorelist, folder, scores_name[0][:-(len("_nr0_scores.npy"))], "Score")
    # for rew in reward_name:
    #     plot_rewards(np.array([np.load(folder + rew) for rew in reward_name]), folder, rew[:-(len("_nr0_reward_curve.npy"))])
    #     plot_rewards(rewardcurve, folder, reward_name[0][:-(len("_nr0_reward_curve.npy"))])