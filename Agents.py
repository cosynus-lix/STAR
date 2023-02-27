import os
import pickle
import json
import numpy as np
import copy
import gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
from keras.metrics import mean_squared_error

import itertools
from collections import defaultdict
import networkx as nx
import argparse
from environment import ndInterval
from dqn_uvf import LowAgent_dqn, HighReplayBuffer
from ddpg_uvf import LowAgent_ddpg
from ddpg import HighAgent, LSTM_HighAgent
from reachability import reach_analysis
from convert import convert

import csv

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--high_batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.99995)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--eps_high', type=float, default=1.0)
parser.add_argument('--eps_high_decay', type=float, default=0.005)
parser.add_argument('--eps_high_min', type=float, default=0.01)
parser.add_argument('--target_update', type=int, default=200)
parser.add_argument('--high_discount_factor', type=int, default=0.99)
parser.add_argument('--reluval_path', type=str, default="/GARA/ReluVal")

args = parser.parse_args()


def listContains(list, element):
    for i in range(len(list)):
        if element in list[i]:
            return i
    return -1


class HRL_Handcrafted:
    """
    This agent uses hand-crafted subgoal partitions. Feudal RL Agent composed of two learning entities. The
    high-level agent decides to navigate to goal sets and rewards the Low-level agent for learning to reach them.
    """

    def __init__(self, env, G_init, h_mem=10000):
        """
        Construct an Feudal RL agent

        :param env: Learning environment
        :param h_mem: High-level memory capacity
        :param l_mem: Low-level memory capacity
        """
        self.env = env
        self.highMemory = HighReplayBuffer(h_mem)
        self.LowAgent = LowAgent_dqn(env)
        self.G = G_init
        self.highLvlPolicy = defaultdict(lambda: np.zeros(len(self.G)))
        self.total_steps = dict()
        self.high_steps = dict()
        self.stats = dict()
        self.automaton = nx.DiGraph()

        for i in range(len(self.G)):
            self.automaton.add_node(i)

    def identify_partition(self, state):
        start_partition = 'none'
        for i in range(len(self.G)):
            if state in self.G[i]:
                start_partition = i
                break
        return start_partition

    def highQ_policy(self, start_partition, epsilon=1):
        Q = self.highLvlPolicy
        policy = make_epsilon_greedy_policy(Q, epsilon, len(self.G))
        action_probs = policy(start_partition)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def highQ_update(self, start_partition, action, reached_partition, reward, done):
        best_next_action = np.argmax(self.highLvlPolicy[reached_partition])
        td_target = reward + (1 - done) * args.high_discount_factor * self.highLvlPolicy[reached_partition][
            best_next_action]
        td_delta = td_target - self.highLvlPolicy[start_partition][action]
        self.highLvlPolicy[start_partition][reached_partition] += args.alpha * td_delta

    def train(self, num_episodes, max_steps, min_steps, k):
        """
        Hierarchical Reinforcement Learning algorithm based on abstract goal representation. High-level agent selects
        goals and Low-level agent uses Q-learning to achieve them .

        Args:
            :param num_episodes: Number of episodes to run for.
            :param max_steps: maximum number of steps in an episode
            :param min_steps: minimum number of steps before analysing policy
            :param k: high-level update frequency
        """

        stats = {"episode_lengths": np.zeros(num_episodes),
                 "episode_rewards": np.zeros(num_episodes),
                 "visits": 1}

        reward_high = 0

        for i_episode in range(num_episodes):
            # Keep track of explored goal transitions during episode
            goals_collection = []

            updateHigh = False

            # Print out which episode we're on, useful for debugging.
            if i_episode % 10 == 0 and i_episode > 0:
                print("\rEpisode {}/{} \t".format(i_episode, num_episodes), end="")
                r = np.mean(stats['episode_rewards'][i_episode - 10: i_episode])
                print("reward : ", r)
                print(epsilon)

            # Reset the environment and pick the first action
            state = self.env.reset()

            for t in itertools.count():

                # Each k steps, the high-level agent selects a goal
                if t % k == 0 or updateHigh:
                    # High-level epsilon annealing
                    epsilon_high -= args.eps_high_decline
                    epsilon_high = max(epsilon_high, args.epsilon_high_min)

                    if t == 0:
                        s_t = state

                        # Determine start goal index
                        start_goal_index = self.identify_partition(s_t)
                    else:
                        reached_state = state
                        reached_partition = self.identify_partition(reached_state)

                        self.highQ_update(start_goal_index, target_goal_index, reached_partition, reward_high, done)
                        # Save high-level transition
                        self.highMemory.put(s_t, start_goal_index, reached_state, target_goal_index, reward_int,
                                            reward_high)
                        s_t = state
                        # Determine start goal index
                        start_goal_index = self.identify_partition(s_t)

                    # Select target partition index
                    target_goal_index = self.highQ_policy(start_goal_index, epsilon_high)
                    goal = np.concatenate([self.G[target_goal_index].inf, self.G[target_goal_index].sup])
                    reward_high = 0
                    updateHigh = False
                    # Add the explored goal transition to collection
                    if (start_goal_index, target_goal_index) not in goals_collection:
                        goals_collection.append((start_goal_index, target_goal_index))

                    # Choose the appropriate Low-level policies or create new ones
                    if (start_goal_index, target_goal_index) in self.high_steps.keys():
                        total_steps = self.total_steps[start_goal_index, target_goal_index]
                        self.high_steps[start_goal_index, target_goal_index] += 1
                    else:
                        total_steps = 0
                        self.high_steps[start_goal_index, target_goal_index] = 1

                # Epsilon decay
                epsilon *= args.eps_decay
                epsilon = max(epsilon, args.epsilon_min)
                # Select action
                action = self.LowAgent.model.get_action(state, goal, epsilon)
                # Take a step
                next_state, reward_ext, done, _ = self.env.step(action)
                # Reward for high-level agent
                reward_high += reward_ext
                reward_int = int(next_state in self.G[target_goal_index])
                reward = reward_int + reward_ext

                # If partition contains reward then assign weight to automaton
                if [start_goal_index, target_goal_index] in self.automaton.edges and reward_ext > 1:
                    self.automaton.add_edge(start_goal_index, target_goal_index, reward=1 / reward_ext)

                # Store transition in low-level memory
                self.LowAgent.buffer.put(state, goal, action, reward, next_state, done)
                # Update statistics
                stats["episode_rewards"][i_episode] += reward_ext
                stats["episode_lengths"][i_episode] = t

                # Policy update
                if self.LowAgent.buffer.size() >= 3000:
                    self.LowAgent.replay()

                # Target update
                if total_steps % args.target_update == 0:
                    self.LowAgent.target_update()

                state = next_state
                total_steps += 1

                self.total_steps[start_goal_index, target_goal_index] = total_steps

                if t > max_steps:
                    break

                if done:
                    print(" Exit reached !")
                    reached_state = state
                    reached_partition = self.identify_partition(reached_state)
                    self.highMemory.put(s_t, start_goal_index, reached_state, target_goal_index, reward_int,
                                        reward_high)
                    self.highQ_update(start_goal_index, target_goal_index, reached_partition, reward_high, done)
                    break

        return stats

    def copy_agent(self, new_agent, low_level=0):
        if low_level == 1:
            Q = self.LowAgent.model.model.get_weights()
            t_Q = self.LowAgent.target_model.model.get_weights()

            new_agent.LowAgent.model.model.set_weights(Q)
            new_agent.LowAgent.target_model.model.set_weights(t_Q)

        return new_agent

    # open the file in the write mode
    def write_partitions(self, path):
        with open(path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            for i in range(len(self.G)):
                # write a row to the csv file
                writer.writerow(self.G[i].inf, self.G[i].sup)
        f.close()

    def read_partitions(self, path):
        G = []
        with open(path, 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # process each line
                G.append(ndInterval(len(line[0]), inf=line[0], sup=line[1]))
        f.close()
        self.G = G


class GARA:
    """
    Feudal RL Agent composed of two learning entities. The high-level agent decides to navigate to goal sets and
    rewards the Low-level agent for learning to reach them.
    """

    def __init__(self, env, G_init, high_level, reachability_algorithm, n=8, h_mem=10000):
        """
        Construct an Feudal RL agent

        :param env: Learning environment
        :param G_init: Initial partitions of states
        :param high_level: high-level strategy Q-learning or Planning
        :param reachability_algorithm: used reachability method Ai2 or ReluVal
        :param n: number of neurons in the 2 layers of the MLP in forward model
        :param h_mem: high-level memory capacity
        """
        self.env = env
        self.reachability_algorithm = reachability_algorithm
        self.state_dim = len(self.env.state)
        self.goal_dim = 2 * self.state_dim
        self.highMemory = HighReplayBuffer(h_mem)
        self.LowAgent = LowAgent_dqn(env)
        self.G = G_init
        self.G_tmp = [{'G': self.G[i], 'R': ndInterval(self.state_dim), 'E': []} for i in
                      range(len(self.G))]  # Temporary partitions list
        self.forward = self.forward_model(n)
        self.strategy = high_level
        if self.strategy == 'Q-learning':
            self.highLvlPolicy = defaultdict(lambda: np.zeros(len(self.G)))

        self.total_steps = defaultdict(lambda: 0)
        self.high_steps = defaultdict(lambda: 0)
        self.automaton = nx.DiGraph()
        self.graph = nx.DiGraph()
        for i in range(len(self.G)):
            self.automaton.add_node(i)
            self.graph.add_node(i)
        self.unsafe = []

    def identify_partition(self, state):
        for i in range(len(self.G)):
            if state in self.G[i]:
                start_partition = i
                break
        return start_partition

    def identify_bounds(self, state, partition):
        l, u = [], []
        in_E = False
        flat_G_tmp = [self.G_tmp[partition]['R']] + self.G_tmp[partition]['E']
        for i in range(len(flat_G_tmp)):
            P = flat_G_tmp[i]
            if state in P:
                l = P.inf
                u = P.sup
                if i > 0:
                    in_E = True
                break
        return l, u, in_E

    def forward_model(self, n, lr=0.005):
        model = tf.keras.Sequential([
            Input((self.state_dim + self.goal_dim,)),
            Dense(n, activation='relu'),
            Dense(n, activation='relu'),
            Dense(self.state_dim)
        ])
        model.compile(loss='mse', optimizer=Adam(lr))
        return model

    def forward_update(self, states, goals, reached_states):
        input = np.concatenate((states, goals), axis=1)
        self.forward.fit(input, reached_states, verbose=0)

    def forward_predict(self, states, goals):
        input = np.concatenate((states, goals), axis=1)
        return self.forward.predict(input, verbose=0)

    def highQ_policy(self, start_partition, epsilon=1):
        Q = self.highLvlPolicy
        policy = make_epsilon_greedy_policy(Q, epsilon, len(self.G))
        action_probs = policy(start_partition)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def highQ_update(self, start_partition, action, reached_partition, reward, done):
        best_next_action = np.argmax(self.highLvlPolicy[reached_partition])
        td_target = reward + (1 - done) * args.high_discount_factor * self.highLvlPolicy[reached_partition][
            best_next_action]
        td_delta = td_target - self.highLvlPolicy[start_partition][action]
        self.highLvlPolicy[start_partition][reached_partition] += args.alpha * td_delta

    def planning(self, start_goal_index, epsilon=1):
        """
        High-level policy that selects goals to achieve. Its selects a successor to start goal with probability
        epsilon, otherwise selects randomly :param start_goal_index:
        :param epsilon:
        :return: selected goal set
        """
        p = np.random.random()
        successors = list(self.graph.successors(start_goal_index))
        if p > epsilon and len(successors) > 1:
            paths = nx.shortest_path_length(self.graph, source=start_goal_index, weight='reward')
            if (start_goal_index, start_goal_index) in self.graph.edges():
                paths[start_goal_index] = nx.get_edge_attributes(self.graph, 'reward')[
                    (start_goal_index, start_goal_index)]

            length = 1
            goal_index = start_goal_index
            for target in paths.keys():
                if paths[target] <= length:
                    length = paths[target]
                    goal_index = target
            path = nx.shortest_path(self.graph, source=start_goal_index, target=goal_index)
            if len(path) > 1:
                target_goal_index = path[1]
            else:
                target_goal_index = path[0]
        else:
            target_goal_index = np.random.choice(len(self.G))

        return target_goal_index

    def split(self, start_partition, target_partition):
        model = self.forward
        input_lower = self.G[start_partition].inf + self.G[target_partition].inf + self.G[target_partition].sup
        input_upper = self.G[start_partition].sup + self.G[target_partition].inf + self.G[target_partition].sup
        output_lower = self.G[target_partition].inf
        output_upper = self.G[target_partition].sup

        model.save("./nnet/forward_" + str(start_partition) + "_" + str(target_partition) + ".h5")
        convert("./nnet/forward_" + str(start_partition) + "_" + str(target_partition) + ".h5", input_lower,
                input_upper)

        # Create json file for property specification
        data = {
            "inputs": {
                "upper": ["{0}".format(str(i)) for i in input_upper],
                "lower": ["{0}".format(str(i)) for i in input_lower]
            },
            "outputs": {
                "upper": ["{0}".format(str(i)) for i in output_upper],
                "lower": ["{0}".format(str(i)) for i in output_lower]
            }
        }
        with open('input.json', 'w') as f:
            json.dump(data, f)

        if self.reachability_algorithm == "Ai2":
            partitions = reach_analysis.reachability_analysis("./nnet/forward_" + str(start_partition) + "_" + str(
                target_partition) + ".nnet", self.G[start_partition], [self.G[target_partition]], "Ai2")

            reach, no_reach = [], []
            for p in partitions['reach']:
                reach += [ndInterval(self.state_dim, p.inf[:self.state_dim], p.sup[:self.state_dim])]
            for p in partitions['no_reach']:
                no_reach += [ndInterval(self.state_dim, p.inf[:self.state_dim], p.sup[:self.state_dim])]

        elif self.reachability_algorithm == "ReluVal":
            reluval_path = args.reluval_path
            os.system(reluval_path + "/network_test  1 ./nnet/forward_" + str(start_partition) + "_" + str(
                target_partition) + ".nnet 0 ./input.json")

            # Saving obtained partitions
            with open('splits.json') as json_file:
                analysis = json.load(json_file)
            reach, no_reach = [], []
            for i in range(len(analysis['reach']['upper'])):
                inf = [float(j) for j in analysis['reach']['lower'][i]]
                sup = [float(j) for j in analysis['reach']['upper'][i]]
                reach += [ndInterval(self.state_dim, inf[:self.state_dim], sup[:self.state_dim])]
            for i in range(len(analysis['no_reach']['upper'])):
                inf = [float(j) for j in analysis['no_reach']['lower'][i]]
                sup = [float(j) for j in analysis['no_reach']['upper'][i]]
                no_reach += [ndInterval(self.state_dim, inf[:self.state_dim], sup[:self.state_dim])]

        reach = ndInterval.search_merge(reach)
        no_reach = ndInterval.search_merge(no_reach)

        self.update_highAgent(start_partition, target_partition, reach, no_reach)

    def update_highAgent(self, start_partition, target_partition, reach, no_reach):
        """
        Update the high_level agents partitions and automaton.
        """
        # Expand the partitions list, update automaton and relabel memory
        next = list(self.automaton.successors(start_partition))
        if reach:
            self.G[start_partition] = copy.deepcopy(reach[0])
            self.automaton.add_edge(start_partition, target_partition, reward=1)
            if len(reach) > 1:
                for i in range(1, len(reach)):
                    self.G.append(copy.deepcopy(reach[i]))
                    if self.strategy == 'Q-learning':
                        self.highLvlPolicy[len(self.G) - 1] = self.highLvlPolicy[start_partition]
                    self.automaton.add_node(len(self.G) - 1)
                    self.automaton.add_edge(len(self.G) - 1, target_partition, reward=1)
                    self.graph.add_node(len(self.G) - 1)

                    for j in next:
                        self.automaton.add_edge(len(self.G) - 1, j, reward=1)
            if no_reach:
                for i in range(len(no_reach)):
                    self.G.append(copy.deepcopy(no_reach[i]))
                    if self.strategy == 'Q-learning':
                        self.highLvlPolicy[len(self.G) - 1] = self.highLvlPolicy[start_partition]
                        self.highLvlPolicy[len(self.G) - 1][target_partition] = 0

                    self.automaton.add_node(len(self.G) - 1)
                    self.graph.add_node(len(self.G) - 1)
                    self.unsafe.append([len(self.G) - 1, target_partition])
                    for j in next:
                        self.automaton.add_edge(len(self.G) - 1, j, reward=1)

            if self.strategy == 'Q-learning':
                for partition in self.highLvlPolicy.keys():
                    tmp = self.highLvlPolicy[partition]
                    self.highLvlPolicy[partition] = np.zeros(len(self.G))
                    self.highLvlPolicy[partition][:len(tmp)] = tmp

            self.relabel()

    # def relabel(self, old_index, new_index):
    #     """
    #     Relables old transitions to respect new goal indexes
    #
    #     :param old_start: old start goal index
    #     :param new_start: new split of start goal index
    #     """
    #     for i in range(len(self.highMemory.buffer)):
    #         if self.highMemory.buffer[i][0] in self.G[new_index]:
    #             self.highMemory.buffer[i][1] = new_index
    #             self.high_steps[new_index, self.highMemory.buffer[i][3]] += 1
    #             self.high_steps[old_index, self.highMemory.buffer[i][3]] -= 1
    #
    #         if self.highMemory.buffer[i][2] in self.G[new_index]:
    #             self.highMemory.buffer[i][3] = new_index
    #             self.high_steps[self.highMemory.buffer[i][1], new_index] += 1
    #             self.high_steps[self.highMemory.buffer[i][1], old_index] -= 1

    def relabel(self):
        """
        Relables old transitions to respect new goal indexes

        :param old_start: old start goal index
        :param new_start: new split of start goal index
        """
        for i in range(len(self.highMemory.buffer)):
            s_t, start_goal_index, reached_state, reward, target_goal_index, reward_high = self.highMemory.buffer[i]
            self.highMemory.put(s_t, self.identify_partition(self.highMemory.buffer[i][0]), reached_state, reward,
                                self.identify_partition(self.highMemory.buffer[i][2]), reward_high)
            self.high_steps[self.highMemory.buffer[i][1], self.highMemory.buffer[i][3]] += 1

        for i in range(len(self.LowAgent.buffer.buffer)):
            state, goal, action, reward, next_state, done = self.LowAgent.buffer.buffer[i]
            goal = self.G[self.identify_partition(next_state)]
            new_goal = np.concatenate([goal.inf, goal.sup])
            self.LowAgent.buffer.put(state, new_goal, action, 1, next_state, done)


    def approximate_partition(self, start_index, target_index):
        """
        Splits starting set in 3 partitions: R containing reaching states, E explored non-reaching and B unexplored states
        """
        sup_R = [0] * self.state_dim
        inf_R = [0] * self.state_dim
        sup_U = [0] * self.state_dim
        inf_U = [0] * self.state_dim
        E = []

        for t in self.highMemory.buffer:
            start_state = t[0]
            reached_state = t[2]
            if start_state in self.G[start_index] and reached_state in self.G[target_index]:
                if sup_R:
                    sup_R = list(np.maximum(sup_R, start_state))
                else:
                    sup_R = start_state
                if inf_R:
                    inf_R = list(np.minimum(inf_R, start_state))
                else:
                    inf_R = start_state

            elif start_state in self.G[start_index] and reached_state not in self.G[start_index]:
                if sup_E:
                    sup_E = list(np.maximum(np.maximum(sup_E, start_state), reached_state))
                else:
                    sup_E = start_state
                if inf_E:
                    inf_E = list(np.minimum(np.minimum(inf_E, start_state), reached_state))
                else:
                    inf_E = start_state
        R = ndInterval(self.state_dim, inf_R, sup_R)
        E = ndInterval(self.state_dim, inf_E, sup_E)

        # R_complement = R.complement(self.G[start_index])
        # for P in R_complement:
        #     intersection = E_tmp.intersection(P)
        #     if intersection:
        #         E.append(intersection)

        return R, E

    def train(self, num_episodes, max_steps, min_steps, k):
        """
        Hierarchical Reinforcement Learning algorithm based on abstract goal representation. High-level agent selects
        goals and Low-level agent uses Q-learning to achieve them .

        Args:
            :param num_episodes: Number of episodes to run for.
            :param max_steps: maximum number of steps in an episode
            :param min_steps: minimum number of steps before analysing policy
            :param k: high-level update frequency
            :param discount_factor: Gamma discount factor.
            :param alpha: TD learning rate.
            :param epsilon: Chance to sample a random action. Float between 0 and 1.
            :param epsilon_min: minimum value for epsilon
            :param eps_decline: decline rate for epsilon)
        """

        stats = {"episode_lengths": np.zeros(num_episodes),
                 "episode_rewards": np.zeros(num_episodes),
                 "internal_rewards": np.zeros(num_episodes),
                 "forward_errors": defaultdict(lambda: np.zeros(num_episodes))}

        reward_high = 0
        epsilon = args.epsilon
        # Random initial split of observable space into two disjoint goal sets

        for i_episode in range(num_episodes):
            # Keep track of explored goal transitions during episode
            goals_collection = []
            prev_target = 0

            # Print out which episode we're on, useful for debugging.
            if i_episode % 10 == 0 and i_episode > 0:
                print("\rEpisode {}/{} \t".format(i_episode, num_episodes), end="")
                r = np.mean(stats['episode_rewards'][i_episode - 10: i_episode])
                print("reward : ", r)
                print("epsilon_low : ", epsilon)
                print("epsilon_high : ", epsilon_high)

            # High-level update monitor
            T = 0
            # Reset the environment and pick the first action
            state = self.env.reset()
            reward_int = 0

            for t in itertools.count():

                # Each k steps, the high-level agent selects a goal
                if t % k == 0:
                    # High-level epsilon annealing
                    epsilon_high -= args.eps_high_decline
                    epsilon_high = max(epsilon_high, args.epsilon_high_min)

                    if t == 0:
                        s_t = state
                        # Determine start goal index
                        start_index = self.identify_partition(s_t)
                    else:
                        prev_target = target_goal_index
                        self.rewards[start_index, target_goal_index].append(rewards)
                        reached_state = state
                        reached_partition = self.identify_partition(reached_state)
                        if self.strategy == 'Q-learning':
                            self.highQ_update(start_goal_index, target_goal_index, reached_partition, reward_high, done)
                        elif self.strategy == 'Planning' and (start_goal_index, reached_partition) not in self.graph.edges():
                            self.graph.add_edge(start_goal_index, reached_partition, reward=1)
                        # Save high-level transition
                        self.highMemory.put(s_t, start_goal_index, reached_state, reward, target_goal_index,
                                            reward_high)
                        if reached_partition != target_goal_index:
                            self.highMemory.put(s_t, start_goal_index, reached_state, reward, reached_partition,
                                                reward_high)

                        s_t = state
                        # Determine start goal index
                        start_goal_index = reached_partition

                    # Select target partition index
                    if self.strategy == 'Q-learning':
                        target_goal_index = self.highQ_policy(start_goal_index, epsilon_high)
                    elif self.strategy == 'Planning':
                        target_goal_index = self.planning(start_goal_index, epsilon_high)

                    goal = np.concatenate([self.G[target_goal_index].inf, self.G[target_goal_index].sup])
                    reward_high = 0

                    # Add the to be explored goal transition to collection
                    if (start_goal_index, target_goal_index) not in goals_collection:
                        goals_collection.append((start_goal_index, target_goal_index))

                    # Choose the appropriate Low-level policies or create new ones
                    if (start_goal_index, target_goal_index) in self.high_steps.keys():
                        total_steps = self.total_steps[start_goal_index, target_goal_index]
                        self.high_steps[start_goal_index, target_goal_index] += 1
                    else:
                        total_steps = 0

                    rewards = 0
                    reached = False

                # Epsilon decay
                epsilon *= args.eps_decay
                # epsilon -= eps_decline
                epsilon = max(epsilon, args.epsilon_min)
                # Select action
                action = self.LowAgent.model.get_action(state, goal, epsilon)
                # Take a step
                next_state, reward_ext, done, _ = self.env.step(action)
                # Reward for high-level agent
                reward_high += reward_ext

                if next_state in self.G[target_goal_index] and reached == False:
                    reached = True
                    reward_int += 1
                    # if start_goal_index != target_goal_index:
                    #     print(start_goal_index, " reached", target_goal_index)

                    if self.strategy == 'Planning' and [start_goal_index, target_goal_index] not in self.unsafe:
                        if reward_ext > 0:
                            self.graph.add_edge(start_goal_index, target_goal_index, reward=1 / reward_ext)
                            if start_goal_index == target_goal_index:
                                indices = list(self.graph.predecessors(target_goal_index))
                                for i in indices:
                                    self.graph.add_edge(i, target_goal_index, reward=1 / reward_ext)
                        elif (start_goal_index, target_goal_index) not in self.graph.edges():
                            self.graph.add_edge(start_goal_index, target_goal_index, reward=1)
                elif next_state in self.G[target_goal_index] and reached == True:
                    reward_int = 0
                else:
                    reward_int = -0.1

                '''
                # Greedy set-based reward
                if state in self.G_tmp[start_goal_index]['R']:
                    reward_int += 0.6
                    reached = True

                # Curiosity reward
                if state in self.G_tmp[start_goal_index]['G'] and state not in self.G_tmp[start_goal_index]['E']:
                    reward_int += 0.3
                    reached = True
                '''
                reward = reward_int + reward_ext

                rewards += reward

                # If partition contains reward then assign weight to automaton
                if [start_goal_index, target_goal_index] in self.automaton.edges and reward_ext > 0:
                    self.automaton.add_edge(start_goal_index, target_goal_index, reward=1 / reward_ext)

                # Store transition in low-level memory
                self.LowAgent.buffer.put(state, goal, action, reward, next_state, done)

                # Store transition in high-level memory
                if t > k and target_goal_index == prev_target:
                    sample = self.LowAgent.buffer.buffer[-k - 1]
                    partition = self.identify_partition(sample[0])
                    self.highMemory.put(sample[0], partition, next_state, prev_target, reward,
                                        reward_high)
                    self.high_steps[self.identify_partition(sample[0]), prev_target] += 1

                # Update statistics
                stats["episode_rewards"][i_episode] += reward_ext
                stats["internal_rewards"][i_episode] += int(reward_int > 0)
                stats["episode_lengths"][i_episode] = t

                # Policy update
                if self.LowAgent.buffer.size() >= 3000:
                    self.LowAgent.replay()

                # Target update
                if total_steps % args.target_update == 0:
                    self.LowAgent.target_update()

                state = next_state
                total_steps += 1

                self.total_steps[start_goal_index, target_goal_index] = total_steps

                if t > max_steps:
                    break

                if done:
                    print(" Exit reached !")
                    reached_state = state
                    reached_partition = self.identify_partition(reached_state)
                    self.highMemory.put(s_t, start_goal_index, reached_state, reward, target_goal_index, reward_high)
                    if reached_partition != target_goal_index:
                        self.highMemory.put(s_t, start_goal_index, reached_state, reward, reached_partition, reward_high)
                    if self.strategy == 'Q-learning':
                        self.highQ_update(start_goal_index, target_goal_index, reached_partition, reward_high, done)
                    elif self.strategy == 'Planning':
                        self.graph.add_edge(start_goal_index, reached_partition, reward=1 / reward_ext)
                        if start_goal_index == reached_partition:
                            indices = list(self.graph.predecessors(reached_partition))
                            for i in indices:
                                if (i, reached_partition) not in self.unsafe:
                                    self.graph.add_edge(i, reached_partition, reward=1 / reward_ext)
                    break

            # Analyse explored goal transitions if the corresponding policy has been trained long and efficiently
            # enough

            for goal_pair in goals_collection:
                if goal_pair not in self.automaton.edges() \
                        and goal_pair not in self.unsafe \
                        and self.high_steps[goal_pair[0], goal_pair[1]] >= min_steps \
                        and goal_pair[0] != goal_pair[1]:
                    # Train forward model
                    states, start_partitions, reached_states, target_partitions, reward_low, reward_high = self.highMemory. \
                        partition_sample(goal_pair[0], goal_pair[1])
                    goals = np.concatenate([self.G[goal_pair[1]].inf, self.G[goal_pair[1]].sup]) * np.ones(
                        shape=(len(states), self.goal_dim))
                    self.forward_update(states=states, goals=goals, reached_states=reached_states)
                    print("Forward model updated")

                    # if model is stable then split
                    states, start_partitions, reached_states, target_partitions, reward_low, reward_high = self.highMemory. \
                        partition_sample(goal_pair[0], goal_pair[1])
                    goals = np.concatenate([self.G[goal_pair[1]].inf, self.G[goal_pair[1]].sup]) * np.ones(
                        shape=(len(states), self.goal_dim))
                    loss = mean_squared_error(reached_states, self.forward_predict(states, goals))
                    stats["forward_errors"][i_episode] = loss
                    if i_episode > 1 and loss[-1] - loss[-2] < 0.01:
                        print("splitting from : ", goal_pair[0], goal_pair[1])
                        self.split(start_partition=goal_pair[0], target_partition=goal_pair[1])
        return stats

    def copy_agent(self, new_agent, low_level=0):

        new_agent.highMemory = copy.deepcopy(self.highMemory)
        new_agent.G = copy.deepcopy(self.G)
        new_agent.total_steps = self.total_steps
        new_agent.high_steps = self.high_steps

        new_agent.automaton = copy.deepcopy(self.automaton)
        new_agent.forward.set_weights(self.forward.get_weights())
        new_agent.unsafe = self.unsafe

        if low_level == 1:
            Q = self.LowAgent.model.model.get_weights()
            t_Q = self.LowAgent.target_model.model.get_weights()

            new_agent.LowAgent.model.model.set_weights(Q)
            new_agent.LowAgent.target_model.model.set_weights(t_Q)

        return new_agent

    def write_partitions(self, path):
        with open(path, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            for i in range(len(self.G)):
                # write a row to the csv file
                writer.writerow(self.G[i].inf + self.G[i].sup)
        f.close()

    def read_partitions(self, path):
        G = []
        with open(path, 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # process each line
                G.append(
                    ndInterval(4, inf=[float(line[i]) for i in range(4)], sup=[float(line[i]) for i in range(4, 8)]))
        f.close()
        self.G = G


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class Feudal_HRL:
    """
    Feudal RL agent without state space abstraction
    """

    def __init__(self, env, representation=0, h_mem=10000, l_mem=10000):
        """
        Construct an Feudal RL agent with no representations
        """
        self.env = env
        self.representation = representation
        # Initialize agent attributes
        if self.representation == 0:
            # No representation
            self.HighAgent = HighAgent(env)
            self.LowAgent = LowAgent_dqn(env, self.representation)
        elif self.representation == 1:
            # LSTM on raw data
            self.HighAgent = LSTM_HighAgent(env)
            self.LowAgent = LowAgent_dqn(env, self.representation)

    def train(self, num_episodes, max_steps, min_steps, k):
        """
        Feudal Hierarchical Reinforcement Learning algorithm based without learning goal representation. High-level agent selects
        goals and Low-level agent uses Q-learning to achieve them .

        """
        stats = {"episode_lengths": np.zeros(num_episodes),
                 "episode_rewards": np.zeros(num_episodes),
                 "visits": 1}
        epsilon_high = args.epsilon_high
        reward_high = 0
        total_steps = 0

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 10 == 0:
                print("\rEpisode {}/{} \t".format(i_episode + 1, num_episodes), end="")
                r = np.mean(stats['episode_rewards'][i_episode + 1 - 10:i_episode])
                # stats["mean_rewards"][(i_episode + 1) // 100] = r
                print("reward : ", r)
                print(epsilon)

            # Reset the environment and pick the first action
            state = self.env.reset()
            bg_noise = np.zeros(self.HighAgent.action_dim)

            for t in itertools.count():
                # Each k steps, the high-level agent selects a goal
                if t % k == 0:
                    # High-level epsilon annealing
                    if epsilon_high > args.epsilon_high_min:
                        epsilon_high -= args.eps_high_decline
                    if t == 0:
                        s_t = state
                        prev_goal = state
                    else:
                        reached_state = state
                        if self.HighAgent.buffer.size() >= 3000:
                            self.HighAgent.replay()
                        # Save high-level transition
                        if self.representation == 0:
                            self.HighAgent.buffer.put(s_t, goal, reward_high, reached_state, done)
                        elif self.representation == 1:
                            self.HighAgent.buffer.put(s_t, prev_goal, goal, reward_high, reached_state, done)
                            prev_goal = goal
                        s_t = state

                    # Select target partition index
                    if self.representation == 0:
                        goal = self.HighAgent.actor.get_action(state)
                    elif self.representation == 1:
                        goal = self.HighAgent.actor.get_action(state, prev_goal)
                    # noise = self.HighAgent.ou_noise(bg_noise, dim=self.HighAgent.action_dim)
                    noise = np.random.randn(self.HighAgent.action_dim) * 0.1
                    goal = np.clip(goal + noise, self.HighAgent.action_bound[0], self.HighAgent.action_bound[1])
                    if (total_steps % k) % args.target_update == 0:
                        self.HighAgent.target_update()

                # Epsilon decay
                epsilon *= args.eps_decay
                epsilon = max(epsilon, args.epsilon_min)
                # Select action
                action = self.LowAgent.model.get_action(state, goal, epsilon)
                # Take a step
                next_state, reward_ext, done, _ = self.env.step(action)
                reward_high += reward_ext
                reward_int = -np.linalg.norm(goal - next_state)

                # Store transition in low-level memory
                self.LowAgent.buffer.put(state, goal, action, reward_int, next_state, done)
                # Update statistics
                stats["episode_rewards"][i_episode] += reward_ext
                stats["episode_lengths"][i_episode] = t

                # Policy update
                if self.LowAgent.buffer.size() >= 3000:
                    self.LowAgent.replay()

                # Target update
                if total_steps % args.target_update == 0:
                    self.LowAgent.target_update()

                if t > max_steps:
                    break

                if done:
                    print(" Exit reached !")
                    reached_state = state
                    if self.representation == 0:
                        self.HighAgent.buffer.put(state, goal, reward_high, next_state, done)
                    elif self.representation == 1:
                        self.HighAgent.buffer.put(state, prev_goal, goal, reward_high, next_state, done)
                    break

                state = next_state
            total_steps += 1

        return stats

    def save(self, file):
        with open(file + 'G.pkl', 'wb') as f:
            pickle.dump(self.G, f)
        with open(file + 'lowagt.pkl', 'wb') as f:
            pickle.dump(self.LowAgent, f)
        self.forward.save(file + 'forward.h5')

    def copy_agent(self, new_agent, low_level=0):
        if self.representation == 1:
            actor_weights = self.HighAgent.actor.model.get_weights()
            t_actor_weights = self.HighAgent.target_actor.model.get_weights()
            critic_weights = self.HighAgent.critic.model.get_weights()
            t_critic_weights = self.HighAgent.target_critic.model.get_weights()

            new_agent.HighAgent.actor.model.set_weights(actor_weights)
            new_agent.HighAgent.critic.model.set_weights(critic_weights)
            new_agent.HighAgent.target_actor.model.set_weights(t_actor_weights)
            new_agent.HighAgent.target_critic.model.set_weights(t_critic_weights)

        if low_level == 1:
            Q = self.LowAgent.model.model.get_weights()
            t_Q = self.LowAgent.target_model.model.get_weights()

            new_agent.LowAgent.model.model.set_weights(Q)
            new_agent.LowAgent.target_model.model.set_weights(t_Q)

        return new_agent
