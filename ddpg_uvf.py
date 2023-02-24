import tensorflow as tf
from keras.layers import Input, Dense, Lambda, concatenate, LSTM

import gym
import argparse
import numpy as np
import random
from collections import deque

# tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--high_batch_size', type=int, default=128)
parser.add_argument('--tau', type=float, default=0.05)
parser.add_argument('--train_start', type=int, default=2000)

args = parser.parse_args()


class HighReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, start_partition, next_state, target_partition, reward_high):
        self.buffer.append([state, start_partition, next_state, target_partition, reward_high])

    def sample(self):
        sample = random.sample(self.buffer, args.high_batch_size)
        states, start_partitions, next_states, target_partitions, reward_high = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, start_partitions, next_states, target_partitions, reward_high

    def partition_sample(self, start_partition, target_partition):
        population = [sample for sample in self.buffer if
                      sample[1] == start_partition and sample[-2] == target_partition]
        sample = random.sample(population, min(len(population),args.high_batch_size))
        states, start_partitions, reached_states, target_partitions, reward_high = map(np.asarray, zip(*sample))
        return states, start_partitions, reached_states, target_partitions, reward_high

    def size(self):
        return len(self.buffer)

class ReplayBuffer:
    def __init__(self, capacity=20000):
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


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, goal_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.goal_dim = goal_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim + self.goal_dim,)),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_dim, activation='tanh'),
            # Lambda(lambda x: x * self.action_bound)
        ])

    def train(self, states, goals, q_grads):
        with tf.GradientTape() as tape:
            grads = tape.gradient(self.model(np.concatenate((states, goals), axis=1)), self.model.trainable_variables, -q_grads)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state, goal):
        return self.model.predict(np.concatenate((state,goal), axis=1), verbose=0)

    def get_action(self, state, goal):
        state = np.reshape(state, [1, self.state_dim])
        goal = np.reshape(goal, [1, self.goal_dim])
        return self.predict(state, goal)[0]


class Critic:
    def __init__(self, state_dim, action_dim, goal_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        state_goal_input = Input((self.state_dim + self.goal_dim,))
        s1 = Dense(16, activation='relu')(state_goal_input)
        s2 = Dense(32, activation='relu')(s1)
        action_input = Input((self.action_dim,))
        a1 = Dense(16, activation='relu')(action_input)
        c1 = concatenate([s2, a1], axis=-1)
        c2 = Dense(32, activation='relu')(c1)
        output = Dense(1, activation='linear')(c2)
        return tf.keras.Model([state_goal_input, action_input], output)

    def predict(self, inputs):
        return self.model.predict(inputs, verbose=0)

    def q_grads(self, states, goals, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([np.concatenate((states, goals), axis=1), actions])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, goals, actions, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model([np.concatenate((states, goals), axis=1), actions], training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class LowAgent_ddpg:
    """
    High level agents that picks states in environment to be visited by low level agents
    """
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.goal_dim = 2 * self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = [-1*np.ones(self.action_dim), np.ones(self.action_dim)]

        self.buffer = ReplayBuffer()

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound[1], self.goal_dim)
        self.critic = Critic(self.state_dim, self.action_dim, self.goal_dim)

        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound[1], self.goal_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim, self.goal_dim)

        actor_weights = self.actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        self.target_actor.model.set_weights(actor_weights)
        self.target_critic.model.set_weights(critic_weights)

    def target_update(self):
        actor_weights = self.actor.model.get_weights()
        t_actor_weights = self.target_actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        t_critic_weights = self.target_critic.model.get_weights()

        for i in range(len(actor_weights)):
            t_actor_weights[i] = args.tau * actor_weights[i] + (1 - args.tau) * t_actor_weights[i]

        for i in range(len(critic_weights)):
            t_critic_weights[i] = args.tau * critic_weights[i] + (1 - args.tau) * t_critic_weights[i]

        self.target_actor.model.set_weights(t_actor_weights)
        self.target_critic.model.set_weights(t_critic_weights)

    def td_target(self, rewards, q_values, dones):
        targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = args.gamma * q_values[i]
        return targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)

    def replay(self):
        states, goals, actions, rewards, next_states, dones = self.buffer.sample()
        target_q_values = self.target_critic.predict([np.concatenate((states, goals), axis=1), self.target_actor.predict(next_states, goals)])
        td_targets = self.td_target(rewards, target_q_values, dones)

        self.critic.train(states, goals, actions, td_targets)

        s_actions = self.actor.predict(states, goals)
        s_grads = self.critic.q_grads(states, goals, s_actions)
        grads = np.array(s_grads).reshape((-1, self.action_dim))
        self.actor.train(states, goals, grads)
        self.target_update()