import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

import numpy as np
import pickle
import os
import json
import copy
from collections import defaultdict

from hrac.models import ControllerActor, ControllerCritic, \
    ManagerActor, ManagerCritic, ForwardModel
from hrac.utils import ndInterval

from reachability import reach_analysis
from convert import convert

import csv


"""
HIRO part adapted from
https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/hiro/hiro.py
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    return tensor.to(device)


def get_tensor(z):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


def make_epsilon_greedy_policy(Q, epsilon, nA, goal=None):
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

    def policy_fn(observation, goal=None):
        if goal is None:
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
        else:
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation, goal])
            A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn

def boltzmann_policy(Q, temperature, nA, goal=None):
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

    def policy_fn(observation, goal=None):
        if goal is None:
            p = np.exp(Q[observation]/temperature).astype('float64')
            action = np.random.choice(4, size=1, p=p/p.sum())
            return int(action[0])    
        else:
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation, goal])
            A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class Boss(object):

    def __init__(self, state_dim, goal_dim, G_init, policy, reachability_algorithm, goal_cond=True, mem_capacity=1e5):
        self.G = G_init
        self.reachability_algorithm = reachability_algorithm
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.policy = policy
        self.partition_steps = defaultdict(lambda: 0)
        self.automaton = nx.DiGraph()
        for i in range(len(self.G)):
            self.automaton.add_node(i)

        if self.policy == 'Q-learning':
            if goal_cond:
                self.Q = np.zeros((len(self.G), len(self.G), len(self.G)))                
            else:
                self.Q = defaultdict(lambda: np.zeros(len(self.G)))
        elif self.policy == 'Planning':
            self.graph = copy.deepcopy(self.automaton)

        self.total_steps = defaultdict(lambda: 0)
        self.high_steps = defaultdict(lambda: 0)
        self.unsafe = []
        

    def identify_partition(self, state):
        """Identify the partition of the state"""
        start_partition = 0
        for i in range(len(self.G)):
            if state in self.G[i]:
                start_partition = i
                break
        return start_partition


    def select_partition(self, start_partition, epsilon, goal=None):
        if goal is not None:
            goal = self.identify_partition(goal)

        if self.policy == 'Q-learning':
            partition = self.Q_learning_policy(start_partition, goal, epsilon)
        elif self.policy == 'Planning':
            partition = self.planning_policy(start_partition, epsilon, goal)
        
        self.partition_steps[start_partition, partition] += 1
        return partition

    def policy_update(self, start_partition, target_partition, reached_partition, reward, done, goal= None, discount=0.99, alpha=0.5):
        if goal is not None:
            goal = self.identify_partition(goal)
        if self.policy == 'Q-learning':
            self.Q_learning_update(start_partition, target_partition, reached_partition, reward, done, discount, alpha, goal)
        elif self.policy == 'Planning':
            self.planning_update(start_partition, target_partition, reached_partition, reward, done)


    def Q_learning_policy(self, start_partition, goal=None, epsilon=1):
        """Epsilon-greedy policy for Q-learning"""
        Q = self.Q
        policy = make_epsilon_greedy_policy(Q, epsilon, len(self.G))
        action_probs = policy(start_partition, goal)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action
    
    def Q_learning_update(self, start_partition, target_partition, reached_partition, reward, done, discount, alpha, goal):
        """Update the Q-table"""
        if goal is None:
            best_next_action = np.argmax(self.Q[reached_partition])
            td_target = reward + (1 - done) * discount * self.Q[reached_partition][
                best_next_action]
            td_delta = td_target - self.Q[start_partition][target_partition]
            self.Q[start_partition][target_partition] += alpha * td_delta
        else:
            best_next_action = np.argmax(self.Q[reached_partition,goal])
            td_target = reward + (1 - done) * discount * self.Q[reached_partition, goal][
                best_next_action]
            td_delta = td_target - self.Q[start_partition, goal][target_partition]
            self.Q[start_partition, goal][target_partition] += alpha * td_delta
            
    def planning_policy(self, start_partition, epsilon=1, goal=None, prev_partition=None):
        """Selects a random partition from the graph with probability epsilon, otherwise selects a successor partition on the shortest path in the graph"""
        p = np.random.random()
        successors = list(self.graph.successors(start_partition))
        
        # if p > epsilon:         
        partition = self.identify_partition(goal)
        if prev_partition is not None and prev_partition in successors:
            successors.remove(prev_partition)
        if start_partition in successors and start_partition != partition:
            successors.remove(start_partition)
        if nx.has_path(self.graph, source=start_partition, target=partition):
            path = nx.shortest_path(self.graph, source=start_partition, target=partition)
            if len(path) > 1:
                target_partition = path[1]
            else:
                target_partition = path[0]
            # elif successors:
            #     target_partition = successors[np.random.randint(0, len(successors))]
        else:
            target_partition = partition

            # elif successors:
            #     paths_len = nx.shortest_path_length(self.graph, source=start_partition, weight='reward')
            #     if (start_partition, start_partition) in self.graph.edges():
            #         paths_len[start_partition] = nx.get_edge_attributes(self.graph, 'reward')[
            #             (start_partition, start_partition)]

            #     length = 1
            #     partition = start_partition
            #     for target in paths_len.keys():
            #         if paths_len[target] <= length:
            #             length = paths_len[target]
            #             partition = target

            #     path = nx.shortest_path(self.graph, source=start_partition, target=partition)
            #     if len(path) > 1:
            #         target_partition = path[1]
            #     else:
            #         target_partition = path[0]
            
            # else:
            #     target_partition = np.random.choice(len(self.G))

        # else:
        #     target_partition = np.random.choice(len(self.G))
        
        return target_partition
    
    def planning_update(self, start_partition, target_partition, reached_partition, reward_ext, done):
        """Update the graph"""
        # if [start_partition, reached_partition] not in self.unsafe:
        #     if (start_partition, reached_partition) in self.graph.edges(): 
        #         reward_list = nx.get_edge_attributes(self.graph, 'reward')
        #         old_reward = reward_list[(start_partition, reached_partition)]
        #         self.graph.add_edge(start_partition, reached_partition, reward=min(abs(reward_ext), old_reward))
        #     else:
        #         self.graph.add_edge(start_partition, reached_partition, reward=abs(reward_ext))
        # elif (start_partition, reached_partition) not in self.graph.edges():
        #     self.graph.add_edge(start_partition, reached_partition, reward=abs(reward_ext))
                

    def split(self, forward_model, start_partition, target_partition, replay_buffer=[]):
        """Split the partition into two partitions according to reachability analysis"""

        model = forward_model

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
            input = ndInterval(len(input_lower), input_lower, input_upper)
            
            partitions = reach_analysis.reachability_analysis("./nnet/forward_" + str(start_partition) + "_" + str(
                target_partition) + ".nnet", input, [self.G[target_partition]], "Ai2")

            reach, no_reach = [], []
            for p in partitions['reach']:
                reach += [ndInterval(self.goal_dim, p.inf[:self.goal_dim], p.sup[:self.goal_dim])]
            for p in partitions['no_reach']:
                no_reach += [ndInterval(self.goal_dim, p.inf[:self.goal_dim], p.sup[:self.goal_dim])]

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

        ndInterval.search_merge(reach)
        ndInterval.search_merge(no_reach)
        
        if reach:
            self.graph_update(start_partition, target_partition, reach, no_reach, replay_buffer)


    def build_graph(self, replay_buffer, reward=-100):
        x, y, sg, u, r, d, _, _ = replay_buffer.sample(len(replay_buffer.storage[0]))
        for i in range(len(x)):
            state, next_state = x[i], y[i]
            start_partition_idx = self.identify_partition(state)
            reached_partition_idx = self.identify_partition(next_state)
            if (start_partition_idx, reached_partition_idx) not in self.graph.edges():
                self.graph.add_edge(start_partition_idx, reached_partition_idx, reward=reward)   

    # def relabel(self, replay_buffer):
    #     """
    #     Relables old transitions to respect new goal indexes

    #     :param old_start: old start goal index
    #     :param new_start: new split of start goal index
    #     """
    #     x, y, sg, u, r, d, _, _ = replay_buffer.sample(len(replay_buffer.storage[0]))
    #     for i in range(len(self.highMemory.buffer)):
    #         s_t, start_goal_index, reached_state, reward, target_goal_index, reward_high = replay_buffer.storage[i]
    #         self.high_steps[self.highMemory.buffer[i][1], self.highMemory.buffer[i][3]] -= 1
    #         self.highMemory.put(s_t, self.identify_partition(self.highMemory.buffer[i][0]), reached_state, reward,
    #                             self.identify_partition(self.highMemory.buffer[i][2]), reward_high)
    #         self.high_steps[self.highMemory.buffer[i][1], self.highMemory.buffer[i][3]] += 1

    #     for i in range(len(self.LowAgent.buffer.buffer)):
    #         state, goal, action, reward, next_state, done = self.LowAgent.buffer.buffer[i]
    #         goal = self.G[self.identify_partition(next_state)]
    #         new_goal = np.concatenate([goal.inf, goal.sup])
    #         self.LowAgent.buffer.put(state, new_goal, action, 1, next_state, done)

   
    def graph_update(self, start_partition, target_partition, reach, no_reach, replay_buffer=[], forward_model= None):
        """
        Update the agent partitions and automaton.
        """
        # Expand the partitions list, update automaton and relabel memory
        discount = 0.99
        n = len(self.G)
        next = list(self.automaton.successors(start_partition))
        if reach:
            if self.policy == 'Q-learning':
                size_G = len(self.G)
                size_reach = len(reach) - 1 
                size_no_reach = len(no_reach)
                print(len(self.G))
                print(self.Q.shape)
                tmp = self.Q[:, :]
                print(tmp.shape)
                self.Q = -100*np.ones((size_G + size_reach + size_no_reach, size_G + size_reach + size_no_reach, size_G + size_reach + size_no_reach))
                print(self.Q.shape)
                self.Q[:size_G, :size_G, :size_G] = tmp

            self.G[start_partition] = copy.deepcopy(reach[0])
            self.automaton.add_edge(start_partition, target_partition, reward=1)
            self.unsafe.append([target_partition, start_partition])
            if self.policy == 'Q-learning':
                tmp = self.Q[start_partition, :, target_partition]
                self.Q[start_partition, :, target_partition] = 0
            elif self.policy == 'Planning' and (start_partition, target_partition) in self.graph.edges:
                reward = 1 # nx.get_edge_attributes(self.graph, 'reward')[(start_partition, target_partition)]
                self.automaton.add_edge(start_partition, target_partition, reward=reward)
            else:
                self.automaton.add_edge(start_partition, target_partition, reward=-10)
                # self.graph.add_edge(start_partition, target_partition, reward=-10)

            if len(reach) > 1:
                for i in range(1, len(reach)):
                    self.G.append(copy.deepcopy(reach[i]))
                    self.automaton.add_node(len(self.G) - 1)
                    self.automaton.add_edge(len(self.G) - 1, target_partition, reward=1)
                    self.unsafe.append([target_partition, len(self.G) - 1 ])
                    if self.policy == 'Q-learning':
                        self.Q[len(self.G) - 1,:,target_partition] = 0
                        self.Q[len(self.G) - 1,:] = self.Q[start_partition,:]
                        self.Q[len(self.G) - 1, :, start_partition] = self.Q[start_partition, :, start_partition]
                        self.Q[start_partition, :, len(self.G) - 1] = self.Q[start_partition, :, start_partition]
                    elif self.policy == 'Planning':
                        self.graph.add_node(len(self.G) - 1)
                        reward = 1 # nx.get_edge_attributes(self.graph, 'reward')[(start_partition, target_partition)]
                        self.graph.add_edge(len(self.G) - 1, target_partition, reward=reward)
                            
                    for j in next:
                        self.automaton.add_edge(len(self.G) - 1, j, reward=-10)
            if no_reach:
                for i in range(len(no_reach)):
                    self.G.append(copy.deepcopy(no_reach[i]))
                    self.automaton.add_node(len(self.G) - 1)
                    self.unsafe.append([len(self.G) - 1, target_partition])
                    for j in next:
                        self.automaton.add_edge(len(self.G) - 1, j, reward=1)

                    if self.policy == 'Q-learning':
                        self.Q[len(self.G) - 1,:] = self.Q[start_partition,:]
                        self.Q[len(self.G) - 1, :][start_partition] = self.Q[start_partition, :][target_partition] / discount
                        self.Q[len(self.G) - 1,:][target_partition] = tmp
                    elif self.policy == 'Planning':
                        self.graph.add_node(len(self.G) - 1)


            
            # elif self.policy == 'Planning':
                # self.build_graph(replay_buffer)

                
    def train(self, forward_model, goal, transition_list, min_steps, batch_size=100, replay_buffer=[]):     
        # goal_partition = self.identify_partition(goal)
        for goal_pair in transition_list:
             if goal_pair not in self.automaton.edges() \
                and goal_pair not in self.unsafe \
                and self.high_steps[(goal_pair[0], 
                                     goal_pair[1])] > min_steps \
                and goal_pair[0] != goal_pair[1] :
                # and not nx.has_path(self.automaton, source=goal_pair[0], target=goal_partition) \
                
                self.split(forward_model, start_partition=goal_pair[0], target_partition=goal_pair[1], replay_buffer=replay_buffer)

    
    def save(self, dir, env_name, algo):
        with open("{}/{}_{}_BossPartitions.pth".format(dir, env_name, algo), 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            for i in range(len(self.G)):
                # write a row to the csv file
                writer.writerow(self.G[i].inf + self.G[i].sup)
        f.close()
        with open("{}/{}_{}_BossPartitions.gpickle".format(dir, env_name, algo), 'wb') as f:
                pickle.dump(self.automaton, f)
    
    def load(self, dir, env_name, algo):
        G = []
        with open("{}/{}_{}_ManagerActor.pth".format(dir, env_name, algo), 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # process each line
                G.append(ndInterval(self.goal_dim, inf=[float(line[i]) for i in range(self.goal_dim // 2)], sup=[float(line[i]) for i in range(self.goal_dim // 2, self.goal_dim)]))
        f.close()
        self.G = G
        
        with open("{}/{}_{}_BossAutomaton.gpickle".format(dir, env_name, algo), 'rb') as f:
            self.automaton = pickle.load(f)
      

class Manager(object):
    def __init__(self, state_dim, goal_dim, action_dim, actor_lr,
                 critic_lr, candidate_goals, correction=True,
                 scale=10, actions_norm_reg=0, policy_noise=0.2,
                 noise_clip=0.5, goal_loss_coeff=0, absolute_goal=False, partitions=False):
        self.scale = scale
        self.actor = ManagerActor(state_dim, goal_dim, action_dim,
                                  scale=scale, absolute_goal=absolute_goal).to(device)
        self.actor_target = ManagerActor(state_dim, goal_dim, action_dim,
                                         scale=scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr, weight_decay=0.0001)

        self.action_norm_reg = 0

        self.criterion = nn.SmoothL1Loss()
        # self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.goal_loss_coeff = goal_loss_coeff
        self.absolute_goal = absolute_goal
        self.partitions = partitions

    def set_eval(self):
        self.actor.set_eval()
        self.actor_target.set_eval()

    def set_train(self):
        self.actor.set_train()
        self.actor_target.set_train()

    def sample_goal(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        if to_numpy:
            return self.actor(state, goal).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, goal).squeeze()

    def value_estimate(self, state, goal, subgoal):
        return self.critic(state, goal, subgoal)

    def actor_loss(self, state, goal, a_net, r_margin):
        actions = self.actor(state, goal)
        eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        if a_net is None and not self.partitions:
            return eval + norm
        elif self.partitions:
            # goal_loss = torch.zeros_like(state[:, :int(self.goal_dim//2)])
            # goal_loss = torch.where(state[:, :int(self.goal_dim//2)] < goal[:, :int(self.goal_dim//2)], torch.square(state[:, :int(self.goal_dim//2)] - goal[:, :int(self.goal_dim//2)]), goal_loss)
            # goal_loss = torch.where(state[:, :int(self.goal_dim//2)] > goal[:, int(self.goal_dim//2) :], torch.square(state[:, :int(self.goal_dim//2)] - goal[:, int(self.goal_dim//2) :]), goal_loss)
            # goal_loss = -torch.sqrt(torch.sum(goal_loss))
            goal_loss = 0
            return eval + norm, goal_loss
        else:
            goal_loss = torch.clamp(F.pairwise_distance(
                a_net(state[:, :self.action_dim]), a_net(state[:, :self.action_dim] + actions)) - r_margin, min=0.).mean()
            return eval + norm, goal_loss

    def off_policy_corrections(self, controller_policy, batch_size, subgoals, x_seq, a_seq):
        first_x = [x[0] for x in x_seq]
        last_x = [x[-1] for x in x_seq]

        # Shape: (batchsz, 1, subgoal_dim)
        diff_goal = (np.array(last_x) - np.array(first_x))[:, np.newaxis, :self.action_dim]

        # Shape: (batchsz, 1, subgoal_dim)
        original_goal = np.array(subgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :self.action_dim],
                                        size=(batch_size, self.candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale[:self.action_dim], self.scale[:self.action_dim])

        # Shape: (batchsz, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # print(np.array(x_seq).shape)
        x_seq = np.array(x_seq)[:, :-1, :]
        a_seq = np.array(a_seq)
        seq_len = len(x_seq[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = a_seq[0][0].shape
        obs_dim = x_seq[0][0].shape
        ncands = candidates.shape[1]

        true_actions = a_seq.reshape((new_batch_sz,) + action_dim)
        observations = x_seq.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            candidate = controller_policy.multi_subgoal_transition(x_seq, candidates[:, c])
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = controller_policy.select_action(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, controller_policy, replay_buffer, iterations, batch_size=100, discount=0.99,
              tau=0.005, a_net=None, r_margin=None):
        avg_act_loss, avg_crit_loss = 0., 0.
        if a_net is not None or self.partitions:
            avg_goal_loss = 0.
        for it in range(iterations):
            # Sample replay buffer
            x, y, g, sgorig, r, d, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            batch_size = min(batch_size, x.shape[0])

            if self.correction and not self.absolute_goal:
                sg = self.off_policy_corrections(controller_policy, batch_size,
                                                 sgorig, xobs_seq, a_seq)
            else:
                sg = sgorig

            state = get_tensor(x)
            next_state = get_tensor(y)
            
            goal = get_tensor(g)
            subgoal = get_tensor(sg)

            reward = get_tensor(r)
            done = get_tensor(1 - d)

            noise = torch.FloatTensor(sgorig).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, goal,
                                          next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(state, goal, subgoal)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                          self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            if a_net is None and not self.partitions:
                actor_loss = self.actor_loss(state, goal, a_net, r_margin)
            elif self.partitions:
                actor_loss, goal_loss = self.actor_loss(state, goal, a_net, r_margin)
                actor_loss = actor_loss + self.goal_loss_coeff * goal_loss
            else:
                actor_loss, goal_loss = self.actor_loss(state, goal, a_net, r_margin)
                actor_loss = actor_loss + self.goal_loss_coeff * goal_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss
            if a_net is not None or self.partitions:
                avg_goal_loss += goal_loss

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if a_net is None and not self.partitions:
            return avg_act_loss / iterations, avg_crit_loss / iterations
        elif self.partitions:
            return avg_act_loss / iterations, avg_crit_loss / iterations, avg_goal_loss / iterations
        else:
            return avg_act_loss / iterations, avg_crit_loss / iterations, avg_goal_loss / iterations

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, dir, env_name, algo):
        torch.save(self.actor.state_dict(), "{}/{}_{}_ManagerActor.pth".format(dir, env_name, algo))
        torch.save(self.critic.state_dict(), "{}/{}_{}_ManagerCritic.pth".format(dir, env_name, algo))
        torch.save(self.actor_target.state_dict(), "{}/{}_{}_ManagerActorTarget.pth".format(dir, env_name, algo))
        torch.save(self.critic_target.state_dict(), "{}/{}_{}_ManagerCriticTarget.pth".format(dir, env_name, algo))
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo):
        self.actor.load_state_dict(torch.load("{}/{}_{}_ManagerActor.pth".format(dir, env_name, algo)))
        self.critic.load_state_dict(torch.load("{}/{}_{}_ManagerCritic.pth".format(dir, env_name, algo)))
        self.actor_target.load_state_dict(torch.load("{}/{}_{}_ManagerActorTarget.pth".format(dir, env_name, algo)))
        self.critic_target.load_state_dict(torch.load("{}/{}_{}_ManagerCriticTarget.pth".format(dir, env_name, algo)))
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo)))


class Controller(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, actor_lr,
                 critic_lr, repr_dim=15, no_xy=True, policy_noise=0.2, noise_clip=0.5,
                 absolute_goal=False
    ):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.no_xy = no_xy
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.absolute_goal = absolute_goal
        self.criterion = nn.SmoothL1Loss()    
        # self.criterion = nn.MSELoss()

        self.actor = ControllerActor(state_dim, goal_dim, action_dim,
                                     scale=max_action).to(device)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim,
                                            scale=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
            lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
            lr=critic_lr, weight_decay=0.0001)


    def clean_obs(self, state, dims=2):
        if self.no_xy:
            with torch.no_grad():
                mask = torch.ones_like(state)
                if len(state.shape) == 3:
                    mask[:, :, :dims] = 0
                elif len(state.shape) == 2:
                    mask[:, :dims] = 0
                elif len(state.shape) == 1:
                    mask[:dims] = 0

                return state*mask
        else:
            return state

    def select_action(self, state, sg, evaluation=False):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        return self.actor(state, sg).cpu().data.numpy().squeeze()

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def actor_loss(self, state, sg):
        return -self.critic.Q1(state, sg, self.actor(state, sg)).mean()

    def subgoal_transition(self, state, subgoal, next_state):
        if self.absolute_goal:
            return subgoal
        else:
            if len(state.shape) == 1:  # check if batched
                return state[:self.goal_dim] + subgoal - next_state[:self.goal_dim]
            else:
                return state[:, :self.goal_dim] + subgoal -\
                       next_state[:, :self.goal_dim]

    def multi_subgoal_transition(self, states, subgoal):
        subgoals = (subgoal + states[:, 0, :self.goal_dim])[:, None] - \
                   states[:, :, :self.goal_dim]
        return subgoals

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        avg_act_loss, avg_crit_loss = 0., 0.
        for it in range(iterations):
            x, y, sg, u, r, d, _, _ = replay_buffer.sample(batch_size)
            next_g = get_tensor(self.subgoal_transition(x, sg, y))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            next_state = self.clean_obs(get_tensor(y))

            noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_g) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, next_g, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, sg, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                          self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = self.actor_loss(state, sg)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations

    def save(self, dir, env_name, algo):
        torch.save(self.actor.state_dict(), "{}/{}_{}_ControllerActor.pth".format(dir, env_name, algo))
        torch.save(self.critic.state_dict(), "{}/{}_{}_ControllerCritic.pth".format(dir, env_name, algo))
        torch.save(self.actor_target.state_dict(), "{}/{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo))
        torch.save(self.critic_target.state_dict(), "{}/{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo))
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo):
        self.actor.load_state_dict(torch.load("{}/{}_{}_ControllerActor.pth".format(dir, env_name, algo)))
        self.critic.load_state_dict(torch.load("{}/{}_{}_ControllerCritic.pth".format(dir, env_name, algo)))
        self.actor_target.load_state_dict(torch.load("{}/{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo)))
        self.critic_target.load_state_dict(torch.load("{}/{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo)))
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo)))
