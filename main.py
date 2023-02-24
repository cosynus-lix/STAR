import os
import pickle
import numpy as np
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d
from Agents import HRL_Handcrafted, GARA, Feudal_HRL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
import matplotlib.pyplot as plt
import argparse
from environment import ObstacleMaze, ndInterval
from draw import representation

parser = argparse.ArgumentParser()


def read_stats(exps):
    exp_list = ['GARA', 'GARA_Planning', 'Handcrafted', 'no_repr', 'LSTM']
    exps = {'GARA': [3, 4, 5], 'GARA_Planning': [1, 3, 4], 'Handcrafted': [2, 3, 4], 'no_repr': [0, 1, 2],
            'LSTM': [0, 1, 2, 2]}
    S = []
    for exp in exp_list:
        stats_list = []
        for i in exps[exp]:
            with open('Labyrinth/EXP_' + exp + '/exp' + str(i) + '.pkl', 'rb') as f:
                stats = pickle.load(f)
                stats_list.append(stats)
        #
        stats = dict()
        for k in stats_list[0].keys():
            matrix = np.array([stats_list[j][k] for j in range(len(stats_list))])
            stats[k] = np.mean(matrix, axis=0)
            # stats[k]['std'] = np.std(matrix, axis=0)

        S.append(stats)

    return S


font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)


def plotting(exp, window=30, xsize=20, ysize=5, metric='r', mode='nearest'):
    exps = {'GARA': [3, 4, 5], 'GARA_Planning': [1, 3, 4], 'Handcrafted': [2, 3, 4], 'no_repr': [0, 1, 2],
            'LSTM': [0, 1, 2, 2, 2]}
    exp_list = ['GARA', 'GARA_Planning', 'Handcrafted', 'no_repr', 'LSTM']
    S = read_stats(exps)

    plt.figure(figsize=(xsize, ysize))

    plt.figure(figsize=(xsize, ysize))

    plt.plot(uniform_filter1d(S[0]["episode_rewards"], size=window, mode=mode), 'tab:green', label='GARA')
    plt.plot(uniform_filter1d(S[1]["episode_rewards"], size=window, mode=mode), 'tab:orange',
             label='GARA-Planning')
    plt.plot(uniform_filter1d(S[2]["episode_rewards"], size=window, mode=mode), 'b',
             label='Handcrafted')
    plt.plot(uniform_filter1d(S[3]["episode_rewards"], size=window, mode=mode), 'y',
             label='Concrete representation')
    plt.plot(uniform_filter1d(S[4]["episode_rewards"], size=window, mode=mode), 'r', label='LSTM')

    plt.xlabel("Episodes")
    plt.ylabel("Mean rewards")
    plt.legend(loc='upper left')
    plt.show()


def statistical_test():
    exp_list = ['base', 'ours', '1', '2']
    test = dict()
    stats_base, stats0, stats1, stats2 = read_stats()

    # Exp 1

    t_stat_base, p_base = stats.ttest_ind(stats0["Base"]["episode_rewards"], stats_base["Base"]["episode_rewards"],
                                          axis=0,
                                          equal_var=True, nan_policy='propagate', permutations=None,
                                          random_state=None,
                                          alternative='two-sided', trim=0)

    t_stat1, p1 = stats.ttest_ind(stats0["Base"]["episode_rewards"], stats1["Base"]["episode_rewards"], axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None, random_state=None,
                                  alternative='two-sided', trim=0)
    t_stat2, p2 = stats.ttest_ind(stats0["Base"]["episode_rewards"], stats2["Base"]["episode_rewards"], axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None,
                                  random_state=None,
                                  alternative='two-sided', trim=0)
    test["Base"] = dict()
    test["Base"]["t-test"] = [t_stat_base, t_stat1, t_stat2]
    test["Base"]["p-value"] = [p_base, p1, p2]

    # Exp 2
    t_stat_base, p_base = stats.ttest_ind(stats0["Complex"]["episode_rewards"],
                                          stats_base["Complex_Base"]["episode_rewards"],
                                          axis=0,
                                          equal_var=True, nan_policy='propagate', permutations=None,
                                          random_state=None,
                                          alternative='two-sided', trim=0)
    t_stat0, p0 = stats.ttest_ind(stats0["Complex"]["episode_rewards"],
                                  stats0["Complex_Base"]["episode_rewards"],
                                  axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None,
                                  random_state=None,
                                  alternative='two-sided', trim=0)

    t_stat1, p1 = stats.ttest_ind(stats0["Complex"]["episode_rewards"], stats1["Complex"]["episode_rewards"], axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None, random_state=None,
                                  alternative='two-sided', trim=0)
    t_stat2, p2 = stats.ttest_ind(stats0["Complex"]["episode_rewards"], stats2["Complex"]["episode_rewards"], axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None,
                                  random_state=None,
                                  alternative='two-sided', trim=0)
    test["Complex"] = dict()
    test["Complex"]["t-test"] = [t_stat_base, t_stat0, t_stat1, t_stat2]
    test["Complex"]["p-value"] = [p_base, p0, p1, p2]

    # Exp 3

    t_stat_base, p_base = stats.ttest_ind(stats0["Points"]["episode_rewards"],
                                          stats_base["Points"]["episode_rewards"],
                                          axis=0,
                                          equal_var=True, nan_policy='propagate', permutations=None,
                                          random_state=None,
                                          alternative='two-sided', trim=0)
    t_stat0, p0 = stats.ttest_ind(stats0["Points"]["episode_rewards"],
                                  stats0["Points_Base"]["episode_rewards"],
                                  axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None,
                                  random_state=None,
                                  alternative='two-sided', trim=0)

    t_stat1, p1 = stats.ttest_ind(stats0["Points"]["episode_rewards"], stats1["Points"]["episode_rewards"], axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None, random_state=None,
                                  alternative='two-sided', trim=0)
    t_stat2, p2 = stats.ttest_ind(stats0["Points"]["episode_rewards"], stats2["Points"]["episode_rewards"], axis=0,
                                  equal_var=True, nan_policy='propagate', permutations=None,
                                  random_state=None,
                                  alternative='two-sided', trim=0)
    test["Points"] = dict()
    test["Points"]["t-test"] = [t_stat_base, t_stat0, t_stat1, t_stat2]
    test["Points"]["p-value"] = [p_base, p0, p1, p2]

    return test


stats = dict()

# U-shaped Maze
walls = [[[1, 1, 0, 2]]]
start = (0, 0)
exit = (0, 3)
env = ObstacleMaze(n=6, start=start, exit=exit, walls=walls)

G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 2, 1, 1]),
                 ndInterval(4, inf=[0, 2, -1, -1], sup=[1, env.n, 1, 1]),
                 ndInterval(4, inf=[1, 2, -1, -1], sup=[env.n, env.n, 1, 1]),
                 ndInterval(4, inf=[1, 0, -1, -1], sup=[env.n, 2, 1, 1])]

agent_Handcrafted = HRL_Handcrafted(env, G_handcrafted)
stats = agent_Handcrafted.train(num_episodes=400, max_steps=300, min_steps=1000, k=5, discount_factor=1,
                                alpha=0.8,
                                epsilon=1, epsilon_high=1,
                                epsilon_min=0.01, epsilon_high_min=0.01, eps_decline=5e-5,
                                eps_high_decline=1e-2)
# 4 Rooms shaped Maze
# Labyrinth Maze

# random.seed(3)
#
# walls_list = [[], [[1, 1, 0, 2]], [[1, 1, 0, 1], [1, 1, 2, 3], [0, 0.5, 1, 1], [1, 1.5, 1, 1]]]
# start_list = [(0, 3), (0, 0), (3, 0)]
# exit_list = [(6, 0), (3, 3), (0, 0)]
#
# walls = [[1, 5, 0.5, 0.5], [1, 1, 0.5, 4], [3, 3, 2.5, 6.1], [5, 5, 0.5, 4]]
# env = ObstacleMaze(n=6, start=start_list[0], exit=exit_list[0], walls=walls)
# G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[3, 3, 1, 1]),
#           ndInterval(4, inf=[0, 3, -1, -1], sup=[3, env.n, 1, 1]),
#           ndInterval(4, inf=[3, 0, -1, -1], sup=[env.n, 3, 1, 1]),
#           ndInterval(4, inf=[3, 3, -1, -1], sup=[env.n, env.n, 1, 1])]
#
# G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 4, 1, 1])]
# G_handcrafted += ndInterval(4, inf=[0, 4, -1, -1], sup=[1, 6, 1, 1]).split([2],lows=[0, 4, -1, -1], ups=[1, 6, 1, 1])
# G_handcrafted += ndInterval(4, inf=[1, 0, -1, -1], sup=[6, 0.5, 1, 1]).split([2],lows=[1, 0, -1, -1], ups=[6, 0.5, 1, 1])
# G_handcrafted += ndInterval(4, inf=[1, 0.5, -1, -1], sup=[3, 6, 1, 1]).split([2,3],lows=[1, 0.5, -1, -1], ups=[3, 6, 1, 1])
# G_handcrafted += ndInterval(4, inf=[3, 0.5, -1, -1], sup=[5.5, 6, 1, 1]).split([2,3],lows=[3, 0.5, -1, -1], ups=[5.5, 6, 1, 1])
# G_handcrafted += ndInterval(4, inf=[5.5, 0, -1, -1], sup=[6, 6, 1, 1]).split([3],lows=[5.5, 0, -1, -1], ups=[6, 6, 1, 1])

# G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[2, 6, 1, 1]),
#           ndInterval(4, inf=[2, 0, -1, -1], sup=[4, 6, 1, 1]),
#           ndInterval(4, inf=[4, 0, -1, -1], sup=[6, 6, 1, 1])]
# G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[3, 2, 1, 1]),
#             ndInterval(4, inf=[3, 0, -1, -1], sup=[6, 2, 1, 1]),
#             ndInterval(4, inf=[0, 2, -1, -1], sup=[3, 6, 1, 1]),
#             ndInterval(4, inf=[3, 2, -1, -1], sup=[6, 6, 1, 1])]
#
# for i in range(5,6):
#     # agent = HRL_Handcrafted(env, G_handcrafted)
#     agent = GARA(env, G_init, 'Planning')
#     stats = agent.train(num_episodes=400, max_steps=300, min_steps=1000, k=10, discount_factor=1,
#                         alpha=0.8,
#                         epsilon=1, epsilon_high=1,
#                         epsilon_min=0.01, epsilon_high_min=0.01, eps_decline=5e-5,
#                         eps_high_decline=1e-2)
# with open('Labyrinth/EXP_GARA_/exp' + str(i) + '.pkl', 'wb') as f:
#     pickle.dump(stats, f)
# agent.write_partitions('Labyrinth/EXP_GARA/exp_repr' + str(i))
# nx.write_gpickle(agent.automaton, 'Labyrinth/EXP_GARA/exp_aut' + str(i) + '.gpickle')

#
# i = 3
# agent = GARA(env, G_init, 'Planning')
# with open:'Labyrinth/EXP_GARA_Planning/exp' + str(i) + '.pkl', 'rb') as f:
#     stats2 = pickle.load(f)
# agent.read_partitions('Labyrinth/EXP_GARA_Planning/exp_repr'+ str(i))
# agent.automaton = nx.read_gpickle('Labyrinth/EXP_GARA_Planning/exp_aut' + str(i) + '.gpickle')
# representation(agent.G, env, 'Labyrinth')
# G = agent.graph.to_undirected()
# indices = list(nx.connected_components(G))[0]
# G = agent.graph.subgraph(indices)
# nx.draw(G1, with_labels=True, font_weight='bold')
# n_exp = 5

# for i in range(0, n_exp):
#
#     walls = [[[1, 5.5, 0.5, 0.5], [1, 1, 0.5, 4], [3, 3, 2.5, 6], [5.5, 5.5, 0.5, 4]]]
#     env = ObstacleMaze(n=6, start=(0,3), exit=(6,0), walls=walls[0])
#     G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 4, 1, 1]),
#                ndInterval(4, inf=[0, 4, -1, -1], sup=[1, 6, 1, 1]),
#                ndInterval(4, inf=[1, 0, -1, -1], sup=[6, 0.25, 1, 1]),
#                ndInterval(4, inf=[1, 0.25, -1, -1], sup=[3, 6, 1, 1]),
#                ndInterval(4, inf=[3, 0.25, -1, -1], sup=[5.5, 6, 1, 1]),
#                ndInterval(4, inf=[5.5, 0, -1, -1], sup=[6, 6, 1, 1])]
#    agent0 = HRL_Handcrafted(env, G_init)
#    agents_base, stats = agent0.evaluate([(3, 0)], [(6,0)], walls)
#    print('Handcrafted Done')
#    with open('Labyrinth/EXP_Handcrafted/exp'+str(i)+'.pkl', 'wb') as f:
#       pickle.dump(stats, f)
#
# G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[3, 3, 1, 1]),
#             ndInterval(4, inf=[3, 0, -1, -1], sup=[6, 3, 1, 1]),
#             ndInterval(4, inf=[0, 3, -1, -1], sup=[3, 6, 1, 1]),
#             ndInterval(4, inf=[3, 3, -1, -1], sup=[6, 6, 1, 1])]
#
# agent = GARA(env, G_init, 'Planning')
# stats = agent.train(num_episodes=400, max_steps=300, min_steps=3000, k=10, discount_factor=1,
#                 alpha=0.8,
#                 epsilon=1, epsilon_high=1,
#                 epsilon_min=0.01, epsilon_high_min=0.01, eps_decline=5e-5,
#                 eps_high_decline=1e-3)
# with open('Labyrinth/EXP_GARA_Planning/exp'+str(i)+'.pkl', 'wb') as f:
#     pickle.dump(stats, f)
# agent.write_partitions('Labyrinth/EXP_GARA_Planning/exp_repr'+str(i)+'.pkl')
# nx.write_gpickle(agent.automaton, 'Labyrinth/EXP_GARA_Planning/exp_aut'+str(i)+'.gpickle')
#
#     agent0 = Feudal_HRL(env, representation=0)
#     agents1, stats1 = agent0.evaluate([(3, 0)], [(6,0)], walls)
#     print('No repr done')
#     with open('Labyrinth/EXP_no_repr/exp'+str(i)+'.pkl', 'wb') as f:
#         pickle.dump(stats0, f)
#
#     agent0 = Feudal_Agent(env, representation=1)
#     agents2, stats2 = agent0.evaluate([(0, 0),(3, 0)], [(3, 0),(0, 0)], walls_list)
#     print('LSTM done')
#     with open('EXP_2/exp'+str(i)+'.pkl', 'wb') as f:
#         pickle.dump(stats0, f)

# plotting(1,30)
# plotting(2,30)
# plotting(3,30)
#
# repr(agents0[0].G, env)


# def plotting(exp, window, xsize=20, ysize=5, metric='r', mode='nearest'):
#     exps = {'GARA': [3, 4, 5], 'GARA_Planning': [3], 'Handcrafted': [0, 1], 'no_repr': [0], 'LSTM': [0]}
#     exp_list = ['GARA', 'GARA_Planning', 'Handcrafted', 'no_repr', 'LSTM']
#     S = read_stats(exps)
#
#     # # Base Training
#
#     if exp == 1:
#         plt.figure(figsize=(xsize, ysize))
#
#         plt.figure(figsize=(xsize, ysize))
#         plt.plot(uniform_filter1d(stats_base["Base"]["episode_rewards"], size=window, mode=mode), 'b',
#                  label='Handcrafted')
#         plt.plot(uniform_filter1d(stats0["Base"]["episode_rewards"], size=window, mode=mode), 'tab:orange',
#                  label='GARA')
#         plt.plot(uniform_filter1d(stats1["Base"]["episode_rewards"], size=window, mode=mode), 'y',
#                  label='Concrete representation')
#         plt.plot(uniform_filter1d(stats2["Base"]["episode_rewards"], size=window, mode=mode), 'r', label='LSTM')
#         # plt.title("U-shaped Maze")
#         plt.xlabel("Episodes")
#         plt.ylabel("Mean rewards")
#         plt.legend(loc='upper left')
#         plt.show()
#
#     elif exp == 2:
#         # Transfer Learning
#         plt.figure(figsize=(xsize, ysize))
#
#         plt.figure(figsize=(xsize, ysize))
#         plt.plot(uniform_filter1d(stats_base["Complex_Base"]["episode_rewards"], size=window, mode=mode),
#                  label='Handcrafted')
#         plt.plot(uniform_filter1d(stats0["Complex_Base"]["episode_rewards"], size=window, mode=mode), 'tab:orange',
#                  label='GARA')
#         plt.plot(uniform_filter1d(stats0["Complex"]["episode_rewards"], size=window, mode=mode), 'lime',
#                  label='GARA - Transfer')
#         plt.plot(uniform_filter1d(stats1["Complex_Base"]["episode_rewards"], size=window, mode=mode), 'y',
#                  label='Concrete representation')
#         plt.plot(uniform_filter1d(stats2["Complex"]["episode_rewards"], size=window, mode=mode), 'tab:brown',
#                  label='Concrete representation - Transfer')
#         plt.plot(uniform_filter1d(stats2["Complex_Base"]["episode_rewards"], size=window, mode=mode), 'r',
#                  label='LSTM')
#         plt.plot(uniform_filter1d(stats1["Complex"]["episode_rewards"], size=window, mode=mode), 'tab:pink',
#                  label='LSTM - Transfer')
#         # plt.title("4 Rooms Maze - Transfer Learning")
#         plt.xlabel("Episodes")
#         plt.ylabel("Mean rewards")
#         plt.legend(loc='upper left')
#         plt.show()
#
#     elif exp == 3:
#         # Start and Exit points switching
#         plt.figure(figsize=(xsize, ysize))
#
#         plt.figure(figsize=(xsize, ysize))
#         plt.plot(uniform_filter1d(stats_base["Points"]["episode_rewards"], size=window, mode=mode),
#                  label='Handcrafted')
#         plt.plot(uniform_filter1d(stats0["Points_Base"]["episode_rewards"], size=window, mode=mode),
#                  'tab:orange', label='GARA')
#         plt.plot(uniform_filter1d(stats0["Points"]["episode_rewards"], size=window, mode=mode), 'lime',
#                  label='GARA - Transfer')
#         plt.plot(uniform_filter1d(stats1["Points_Base"]["episode_rewards"], size=window, mode=mode), 'y',
#                  label='Concrete representation')
#         plt.plot(uniform_filter1d(stats1["Points"]["episode_rewards"], size=window, mode=mode), 'tab:brown',
#                  label='Concrete representation - Transfer')
#         plt.plot(uniform_filter1d(stats2["Points_Base"]["episode_rewards"], size=window, mode=mode), 'r',
#                  label='LSTM')
#         plt.plot(uniform_filter1d(stats2["Points"]["episode_rewards"], size=window, mode=mode), 'tab:pink',
#                  label='LSTM - Transfer')
#         # plt.title("U-shaped Maze - Switched START and EXIT")
#         plt.xlabel("Episodes")
#         plt.ylabel("Mean rewards")
#         plt.legend(loc='upper left')
#         plt.show()

G = ndInterval(2, [0, 0], [10, 10])
subinterval = ndInterval(2, [3, 3], [7, 7])
complement = G.complement(subinterval)
print(len(complement))
for i in complement:
    print(i.inf, i.sup, "\n")

merged = ndInterval.search_merge(complement + [subinterval])
print('merged')
for i in merged:
    print(i.inf, i.sup, "\n")
