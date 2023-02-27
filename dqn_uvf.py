import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam

import argparse
import numpy as np
from collections import deque
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--high_batch_size', type=int, default=32)


args = parser.parse_args()

class HighReplayBuffer_augmented:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, start_partition, next_state, target_partition, reward_low, reward_high, l, u):
        self.buffer.append([state, start_partition, next_state, target_partition, reward_low, reward_high, l, u])

    def sample(self):
        sample = random.sample(self.buffer, args.high_batch_size)
        states, start_partitions, next_states, target_partitions, reward_low, reward_high, l, u = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        l = np.array(l).reshape(args.batch_size, -1)
        u = np.array(u).reshape(args.batch_size, -1)
        return states, start_partitions, next_states, target_partitions, reward_low, reward_high, l, u

    def partition_sample(self, start_partition, target_partition):
        population = [sample for sample in self.buffer if
                      sample[1] == start_partition and sample[3] == target_partition]
        sample = random.sample(population, min(len(population), args.high_batch_size))
        states, start_partitions, reached_states, target_partitions, reward_low, reward_high, l, u = map(np.asarray, zip(*sample))
        return states, start_partitions, reached_states, target_partitions, reward_low, reward_high, l, u

    def size(self):
        return len(self.buffer)


class HighReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, start_partition, next_state, target_partition, reward_low, reward_high):
        self.buffer.append([state, start_partition, next_state, target_partition, reward_low, reward_high])

    def sample(self):
        sample = random.sample(self.buffer, args.high_batch_size)
        states, start_partitions, next_states, target_partitions, reward_low, reward_high = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, start_partitions, next_states, target_partitions, reward_low, reward_high

    def partition_sample(self, start_partition, target_partition):
        population = [sample for sample in self.buffer if
                      sample[1] == start_partition and sample[3] == target_partition]
        sample = random.sample(population, min(len(population), args.high_batch_size))
        states, start_partitions, reached_states, target_partitions, reward_low, reward_high = map(np.asarray, zip(*sample))
        #states = np.array(states).reshape(args.batch_size, -1)
        #reached_states = np.array(reached_states).reshape(args.batch_size, -1)
        return states, start_partitions, reached_states, target_partitions, reward_low, reward_high

    def rewarding_partition_sample(self, start_partition, target_partition):
        population = [sample for sample in self.buffer if
                      sample[1] == start_partition and sample[3] == target_partition and sample[-2]>0]
        states, start_partitions, reached_states, target_partitions, reward_low, reward_high = map(np.asarray, zip(*population))
        return states, start_partitions, reached_states, target_partitions, reward_low, reward_high

    def size(self):
        return len(self.buffer)


class ReplayBuffer_uvf:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, goal, action, reward, next_state, done):
        self.buffer.append([state, goal, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, goals, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        goals = np.array(goals).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, goals, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class Qnet_uvf:
    def __init__(self, state_dim, goal_dim, action_dim):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim + self.goal_dim,)),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse', optimizer=Adam(args.lr))
        return model

    def predict(self, input):
        return self.model.predict(input, verbose=0)

    def get_action(self, state, goal, epsilon):
        state = np.reshape(state, [1, self.state_dim])
        goal = np.reshape(goal, [1, self.goal_dim])
        input = tf.convert_to_tensor(np.concatenate((state, goal), axis=1))
        q_value = self.predict(input)[0]
        if np.random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, goals, targets):
        self.model.fit(np.concatenate((states, goals), axis = 1), targets, epochs=1, verbose=0)


class LowAgent_dqn:
    def __init__(self, env, abstraction=True):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        if abstraction:
            self.goal_dim = 2*self.state_dim
        else:
            self.goal_dim = self.state_dim
        self.action_dim = self.env.action_space.n

        self.model = Qnet_uvf(self.state_dim, self.goal_dim, self.action_dim)
        self.target_model = Qnet_uvf(self.state_dim, self.goal_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer_uvf()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        states, goals, actions, rewards, next_states, done = self.buffer.sample()
        inputs = tf.convert_to_tensor(np.concatenate((states, goals), axis=1))
        targets = self.target_model.predict(inputs)
        inputs = tf.convert_to_tensor(np.concatenate((next_states, goals), axis=1))
        next_q_values = self.target_model.predict(inputs).max(axis=1)
        targets[range(args.batch_size), actions] = np.clip(rewards + (1 - done) * next_q_values * args.gamma, -1, 1)
        self.model.train(states, goals, targets)
