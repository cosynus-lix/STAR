import os
import json
import pickle
import numpy as np
import networkx as nx
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d
from Agents import HRL_Handcrafted, GARA, Feudal_HRL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
import matplotlib.pyplot as plt
import argparse
from environment import ObstacleMaze, ndInterval
from draw import representation
from utils import read_stats, plotting, statistical_test

parser = argparse.ArgumentParser()

font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

stats = dict()

# U-shaped Maze
walls = [[2, 4, -1, 4]]
start = (0, 0)
exit = (6, 0)
env = ObstacleMaze(n=6, start=start, exit=exit, walls=walls)

for exp in range(10):
    G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[2, 4, 1, 1]),
                     ndInterval(4, inf=[0, 4, -1, -1], sup=[2, env.n, 1, 1]),
                     ndInterval(4, inf=[2, 4, -1, -1], sup=[4, env.n, 1, 1]),
                     ndInterval(4, inf=[2, 0, -1, -1], sup=[4, 4, 1, 1]),
                     ndInterval(4, inf=[4, 4, -1, -1], sup=[env.n, env.n, 1, 1]),
                     ndInterval(4, inf=[4, 0, -1, -1], sup=[env.n, 4, 1, 1])]

    agent_Handcrafted = HRL_Handcrafted(env, G_handcrafted)
    stats['Handcrafted'] = agent_Handcrafted.train(num_episodes=300, max_steps=300, min_steps=1000, k=5)

    G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[3, env.n, 1, 1]),
              ndInterval(4, inf=[3, 0, -1, -1], sup=[env.n, env.n, 1, 1])]

    agent_GARA_Q = GARA(env, G_init, 'Q-learning', 'Ai2')
    stats['GARA_Q-learning'] = agent_GARA_Q.train(num_episodes=300, max_steps=300, min_steps=1000, k=5)

    agent_GARA_Q.write_partitions('U-shaped/Q_part_exp' + str(exp))
    with open('U-shaped/Q_graph_exp' + str(exp) + '.gpickle', 'wb') as f:
        pickle.dump(agent_GARA_Q.automaton, f)

    agent_GARA_P = GARA(env, G_init, 'Planning', 'Ai2')
    stats['GARA_Planning'] = agent_GARA_P.train(num_episodes=1, max_steps=300, min_steps=1000, k=5)

    agent_GARA_P.write_partitions('U-shaped/P_part_exp' + str(exp))
    with open('U-shaped/P_graph_exp' + str(exp) + '.gpickle', 'wb') as f:
        pickle.dump(agent_GARA_P.automaton, f)

    agent_HDQN = Feudal_HRL(env, representation=0)
    stats['HDQN'] = agent_HDQN.train(num_episodes=300, max_steps=300, min_steps=1000, k=5)

    agent_LSTM = Feudal_HRL(env, representation=1)
    stats['LSTM'] = agent_LSTM.train(num_episodes=300, max_steps=300, min_steps=1000, k=5)

    with open('U-shaped/exp' + str(exp) + '.pkl', 'wb') as f:
        pickle.dump(stats, f)

    print('U-shaped done')

# with open('U-shaped.json', 'w') as f:
#     json.dump(stats, f)

# 4 Rooms shaped Maze
'''
walls = [[1, 1, 0, 1], [1, 1, 2, 3], [0, 0.5, 1, 1], [1, 1.5, 1, 1]]
start = (0, 0)
exit = (0, 3)
env = ObstacleMaze(n=3, start=start, exit=exit, walls=walls)

G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 2, 1, 1]),
                 ndInterval(4, inf=[0, 2, -1, -1], sup=[1, env.n, 1, 1]),
                 ndInterval(4, inf=[1, 2, -1, -1], sup=[env.n, env.n, 1, 1]),
                 ndInterval(4, inf=[1, 0, -1, -1], sup=[env.n, 2, 1, 1])]

agent_Handcrafted = HRL_Handcrafted(env, G_handcrafted)
stats['Handcrafted'] = agent_Handcrafted.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, env.n, 1, 1]),
          ndInterval(4, inf=[1, 0, -1, -1], sup=[env.n, env.n, 1, 1])]

agent_GARA_transfer = agent_GARA.copy_agent()
stats['GARA'] = agent_GARA_transfer.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

agent_GARA = GARA(env, G_init, 'Planning', 'Ai2')
stats['GARA'] = agent_GARA.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

agent_HDQN = Feudal_HRL(env, representation=0)
stats['HDQN'] = agent_HDQN.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

agent_LSTM = Feudal_HRL(env, representation=1)
stats['LSTM'] = agent_LSTM.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

# with open('4Rooms.json', 'w') as f:
#     json.dump(stats, f)

# Labyrinth Maze
walls = [[1, 5, 0.5, 0.5], [1, 1, 0.5, 4], [3, 3, 2.5, 6.1], [5, 5, 0.5, 4]]
start = (0, 0)
exit = (0, 6)
env = ObstacleMaze(n=6, start=start, exit=exit, walls=walls)

G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 4, 1, 1])]
G_handcrafted += ndInterval(4, inf=[0, 4, -1, -1], sup=[1, 6, 1, 1]).split([2],lows=[0, 4, -1, -1], ups=[1, 6, 1, 1])
G_handcrafted += ndInterval(4, inf=[1, 0, -1, -1], sup=[6, 0.5, 1, 1]).split([2],lows=[1, 0, -1, -1], ups=[6, 0.5, 1, 1])
G_handcrafted += ndInterval(4, inf=[1, 0.5, -1, -1], sup=[3, 6, 1, 1]).split([2,3],lows=[1, 0.5, -1, -1], ups=[3, 6, 1, 1])
G_handcrafted += ndInterval(4, inf=[3, 0.5, -1, -1], sup=[5.5, 6, 1, 1]).split([2,3],lows=[3, 0.5, -1, -1], ups=[5.5, 6, 1, 1])
G_handcrafted += ndInterval(4, inf=[5.5, 0, -1, -1], sup=[6, 6, 1, 1]).split([3],lows=[5.5, 0, -1, -1], ups=[6, 6, 1, 1])

agent_Handcrafted = HRL_Handcrafted(env, G_handcrafted)
stats['Handcrafted'] = agent_Handcrafted.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[3, 3, 1, 1]),
          ndInterval(4, inf=[0, 3, -1, -1], sup=[3, env.n, 1, 1]),
          ndInterval(4, inf=[3, 0, -1, -1], sup=[env.n, 3, 1, 1]),
          ndInterval(4, inf=[3, 3, -1, -1], sup=[env.n, env.n, 1, 1])]

agent_GARA = GARA(env, G_init, 'Planning', 'Ai2')
stats['GARA'] = agent_GARA.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

agent_HDQN = Feudal_HRL(env, representation=0)
stats['HDQN'] = agent_HDQN.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

agent_LSTM = Feudal_HRL(env, representation=1)
stats['LSTM'] = agent_LSTM.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

# with open('Labyrinth.json', 'w') as f:
#     json.dump(stats, f)
'''