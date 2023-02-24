import os
import pickle
import numpy as np
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()


def read_stats():
    exp_list = ['GARA', 'GARA_Planning', 'Handcrafted', 'no_repr', 'LSTM']
    exps = {'GARA': [3, 4, 5], 'GARA_Planning': [1, 3, 4], 'Handcrafted': [2, 3, 4], 'no_repr': [0,1,2], 'LSTM': [0, 1, 2,2]}
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
    exps = {'GARA': [3, 4, 5], 'GARA_Planning': [1, 3, 4], 'Handcrafted': [2, 3, 4], 'no_repr': [0, 1, 2], 'LSTM': [0, 1, 2, 2, 2]}
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

