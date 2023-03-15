import os
import json
import pickle
import csv
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
walls = [[1, 1.5, -1, 2]]
start = (0, 0)
exit = (3, 0)
env = ObstacleMaze(n=3, start=start, exit=exit, walls=walls)

for exp in range(10):

    G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 2, 1, 1]),
                    ndInterval(4, inf=[0, 2, -1, -1], sup=[1, env.n, 1, 1]),
                    ndInterval(4, inf=[1, 2, -1, -1], sup=[1.5, env.n, 1, 1]),
                    ndInterval(4, inf=[1, 0, -1, -1], sup=[1.5, 2, 1, 1]),
                    ndInterval(4, inf=[1.5, 2, -1, -1], sup=[env.n, env.n, 1, 1]),
                    ndInterval(4, inf=[1.5, 0, -1, -1], sup=[env.n, 2, 1, 1])]

    agent_Handcrafted = HRL_Handcrafted(env, G_handcrafted)
    stats['Handcrafted'] = agent_Handcrafted.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

    with open('U-shaped/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, env.n, 1, 1]),
            ndInterval(4, inf=[1, 0, -1, -1], sup=[env.n, env.n, 1, 1])]

    agent_GARA_Q = GARA(env, G_init, 'Q-learning', 'Ai2')
    stats['GARA_Q-learning'] = agent_GARA_Q.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

    with open('U-shaped/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    agent_GARA_Q.write_partitions('U-shaped/Q_part_exp' + str(exp))
    with open('U-shaped/Q_graph_exp' + str(exp) + '.gpickle', 'wb') as f:
            pickle.dump(agent_GARA_Q.automaton, f)

    agent_GARA_P = GARA(env, G_init, 'Planning', 'Ai2')
    stats['GARA_Planning'] = agent_GARA_P.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

    with open('U-shaped/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    agent_GARA_P.write_partitions('U-shaped/P_part_exp' + str(exp))
    with open('U-shaped/P_graph_exp' + str(exp) + '.gpickle', 'wb') as f:
            pickle.dump(agent_GARA_P.automaton, f)

    agent_HDQN = Feudal_HRL(env, representation=0)
    stats['HDQN'] = agent_HDQN.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

    with open('U-shaped/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    agent_LSTM = Feudal_HRL(env, representation=1)
    stats['LSTM'] = agent_LSTM.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

    with open('U-shaped/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    print('U-shaped done')

# 4 Rooms shaped Maze

walls = [[1, 1.5, -1, 1], [1, 1.5, 2, 4], [-1, 0.5, 1, 1.5], [1, 1.5, 1, 1.5]]
start = (0, 0)
exit = (3, 0)
env = ObstacleMaze(n=3, start=start, exit=exit, walls=walls)


for exp in range(1,5):


        G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 2, 1, 1]),
                 ndInterval(4, inf=[0, 2, -1, -1], sup=[1, env.n, 1, 1]),
                 ndInterval(4, inf=[1, 2, -1, -1], sup=[env.n, env.n, 1, 1]),
                 ndInterval(4, inf=[1, 0, -1, -1], sup=[env.n, 2, 1, 1])]

        agent_Handcrafted = HRL_Handcrafted(env, G_handcrafted)
        stats['Handcrafted'] = agent_Handcrafted.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

        with open('4Rooms/exp' + str(exp) + '.pkl', 'wb') as f:
                pickle.dump(stats, f)

        G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, env.n, 1, 1]),
                ndInterval(4, inf=[1, 0, -1, -1], sup=[env.n, env.n, 1, 1])]

        agent_GARA_Q = GARA(env, G_init, 'Q-learning', 'Ai2')
        stats['GARA_Q-learning'] = agent_GARA_Q.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

        with open('4Rooms/exp' + str(exp) + '.pkl', 'wb') as f:
                pickle.dump(stats, f)

        agent_GARA_Q.write_partitions('4Rooms/Q_part_exp' + str(exp))
        with open('4Rooms/Q_graph_exp' + str(exp) + '.gpickle', 'wb') as f:
                pickle.dump(agent_GARA_Q.automaton, f)

        G_init_transfer = []
        with open('U-shaped/Q_part_exp' + str(exp), 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # process each line
                G_init_transfer.append(ndInterval(4, inf=[float(line[i]) for i in range(4)], sup=[float(line[i]) for i in range(4, 8)]))

        agent_GARA_Q_transfer = GARA(env, G_init_transfer, 'Q-learning', 'Ai2')
        stats['GARA_Q-learning_transfer'] = agent_GARA_Q_transfer.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

        with open('4Rooms/exp' + str(exp) + '.pkl', 'wb') as f:
                pickle.dump(stats, f)

        agent_GARA_Q_transfer.write_partitions('4Rooms/Q_part_transfer_exp' + str(exp))
        with open('4Rooms/Q_graph_transfer_exp' + str(exp) + '.gpickle', 'wb') as f:
                pickle.dump(agent_GARA_Q_transfer.automaton, f)


        agent_GARA_P = GARA(env, G_init, 'Planning', 'Ai2')
        stats['GARA_Planning'] = agent_GARA_P.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

        with open('4Rooms/exp' + str(exp) + '.pkl', 'wb') as f:
                pickle.dump(stats, f)

        agent_GARA_P.write_partitions('4Rooms/P_part_exp' + str(exp))
        with open('4Rooms/P_graph_exp' + str(exp) + '.gpickle', 'wb') as f:
                pickle.dump(agent_GARA_P.automaton, f)

        G_init_transfer = []
        with open('U-shaped/P_part_exp' + str(exp), 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # process each line
                G_init_transfer.append(ndInterval(4, inf=[float(line[i]) for i in range(4)], sup=[float(line[i]) for i in range(4, 8)]))

        agent_GARA_P_transfer = GARA(env, G_init_transfer, 'Planning', 'Ai2')
        agent_GARA_P_transfer.read_partitions('U-shaped/P_part_exp' + str(exp))
        stats['GARA_Planning_transfer'] = agent_GARA_P_transfer.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

        with open('4Rooms/exp' + str(exp) + '.pkl', 'wb') as f:
                pickle.dump(stats, f)

        agent_GARA_P_transfer.write_partitions('4Rooms/P_part_transfer_exp' + str(exp))
        with open('4Rooms/P_graph_transfer_exp' + str(exp) + '.gpickle', 'wb') as f:
                pickle.dump(agent_GARA_P_transfer.automaton, f)

        agent_HDQN = Feudal_HRL(env, representation=0)
        stats['HDQN'] = agent_HDQN.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

        with open('4Rooms/exp' + str(exp) + '.pkl', 'wb') as f:
                pickle.dump(stats, f)

        agent_LSTM = Feudal_HRL(env, representation=1)
        stats['LSTM'] = agent_LSTM.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

        with open('4Rooms/exp' + str(exp) + '.pkl', 'wb') as f:
                pickle.dump(stats, f)

        print('4Rooms done')

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


walls = [[1, 1.5, -1, 2], [4, 4.5, 1, 4], [-1, 7, 3, 4]]
start = (0, 0)
exit = (6, 3)
env = ObstacleMaze(n=6, start=start, exit=exit, walls=walls)


def scooch(G, distance, dimension):
    for i in range(len(G)):
        inf = G[i].inf
        sup = G[i].sup
        inf[dimension] += distance
        sup[dimension] += distance
        G[i] = ndInterval(len(inf), inf, sup)


for exp in range(2):
    with open('BigMaze/exp' + str(exp) + '.pkl', 'rb') as f:
        stats = pickle.load(f)

    G_handcrafted = [ndInterval(4, inf=[0, 0, -1, -1], sup=[1, 2, 1, 1]),
            ndInterval(4, inf=[0, 2, -1, -1], sup=[1, 3, 1, 1]),
            ndInterval(4, inf=[1, 0, -1, -1], sup=[4, 2, 1, 1]),
            ndInterval(4, inf=[1, 2, -1, -1], sup=[4, 3, 1, 1]),
            ndInterval(4, inf=[4, 0, -1, -1], sup=[env.n, 2, 1, 1]),
            ndInterval(4, inf=[4, 2, -1, -1], sup=[env.n, 3, 1, 1])]

    agent_Handcrafted = HRL_Handcrafted(env, G_handcrafted)
    stats['Handcrafted'] = agent_Handcrafted.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

    with open('BigMaze/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    G_init = [ndInterval(4, inf=[0, 0, -1, -1], sup=[3, 3, 1, 1]),
            ndInterval(4, inf=[3, 0, -1, -1], sup=[env.n, 3, 1, 1])]

    agent_GARA_Q = GARA(env, G_init, 'Q-learning', 'Ai2')
    stats['GARA_Q-learning'] = agent_GARA_Q.train(num_episodes=400, max_steps=300, min_steps=5000, k=5)

    with open('BigMaze/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    agent_GARA_Q.write_partitions('BigMaze/Q_part_exp' + str(exp))
    with open('BigMaze/Q_graph_exp' + str(exp) + '.gpickle', 'wb') as f:
            pickle.dump(agent_GARA_Q.automaton, f)

    G_init_transfer = []
    with open('U-shaped/P_part_exp' + str(0), 'r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            # process each line
            G_init_transfer.append(ndInterval(4, inf=[float(line[i]) for i in range(4)], sup=[float(line[i]) for i in range(4, 8)]))

    scooch(G_init_transfer, 3, 0)
    G_init_transfer += [ndInterval(4, inf=[0, 0, -1, -1], sup=[3, 3, 1, 1])]

    agent_GARA_Q_transfer = GARA(env, G_init_transfer, 'Q-learning', 'Ai2')
    stats['GARA_Q-learning_transfer'] = agent_GARA_Q_transfer.train(num_episodes=400, max_steps=300, min_steps=10000, k=5)

    with open('BigMaze/exp' + str(exp) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    agent_GARA_Q_transfer.write_partitions('BigMaze/Q_part_transfer_exp' + str(exp))
    with open('BigMaze/Q_graph_transfer_exp' + str(exp) + '.gpickle', 'wb') as f:
            pickle.dump(agent_GARA_Q_transfer.automaton, f)

    agent_HDQN = Feudal_HRL(env, representation=0)
    stats['HDQN'] = agent_HDQN.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

    with open('BigMaze/exp' + str(exp) + '.pkl', 'wb') as f:
        pickle.dump(stats, f)

    agent_LSTM = Feudal_HRL(env, representation=1)
    stats['LSTM'] = agent_LSTM.train(num_episodes=400, max_steps=300, min_steps=1000, k=5)

    with open('BigMaze/exp' + str(exp) + '.pkl', 'wb') as f:
        pickle.dump(stats, f)

    print('Bigger Maze done')
