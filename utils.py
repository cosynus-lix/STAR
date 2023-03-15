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

font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

def read_stats(setup):

    stats_list = []

    for i in range(8):
        with open('./' + setup + '/exp' + str(i) + '.pkl', 'rb') as f:
            stats = pickle.load(f)
            stats_list.append(stats)

    stats = dict()
    for alg in stats_list[0].keys():
        matrix = np.array([stats_list[j][alg]['episode_rewards'] for j in range(len(stats_list))])
        stats[alg] = np.mean(matrix, axis=0)
        # stats[k]['std'] = np.std(matrix, axis=0)

    return stats


font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)


def plotting(setup, window=30, xsize=20, ysize=5, metric='r', mode='nearest'):
    stats = read_stats(setup)

    plt.figure(figsize=(xsize, ysize))

    plt.plot(uniform_filter1d(stats['Handcrafted'], size=window, mode=mode), 'b', label='Handcrafted')
    plt.plot(uniform_filter1d(stats['GARA_Q-learning'], size=window, mode=mode), 'tab:orange', label='GARA')
    # plt.plot(uniform_filter1d(stats['GARA_Planning'], size=window, mode=mode),
    #          label='GARA-Planning')
    plt.plot(uniform_filter1d(stats['GARA_Q-learning_transfer'], size=window, mode=mode), 'lime', label='GARA - Transfer')
    plt.plot(uniform_filter1d(stats['HDQN'], size=window, mode=mode), 'y', label='Concrete representation')
    plt.plot(uniform_filter1d(stats['LSTM'], size=window, mode=mode), 'r', label='LSTM')
    plt.plot(uniform_filter1d(stats['HDQN_transfer'], size=window, mode=mode), 'tab:brown', label='Concrete representation - Transfer')
    plt.plot(uniform_filter1d(stats['LSTM_transfer'], size=window, mode=mode), 'tab:pink', label='LSTM - Transfer')


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

