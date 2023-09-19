import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import copy
from collections import deque
from interval import interval


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Multi dimensional intervals class
class ndInterval:
    """
    Class that creates arrays of intervals and extend interval methods across the array.
    """

    def __init__(self, n, inf=[], sup=[]):
        self.n = n
        self.inf = inf
        self.sup = sup
        if inf != [] and sup != []:
            self.interval = [interval[inf[i], sup[i]] for i in range(n)]
        else:
            self.interval = []

    def __contains__(self, item):
        # assert self.n == len(item)
        for i in range(self.n):
            if not item[i] in self.interval[i]:
                return False
        return True

    def volume(self):
        volume = 1
        for i in range(self.n):
            volume *= self.sup[i] - self.inf[i]

        return volume

    def adjacency(self, B):
        """Checks for adjacent intervals that can be merged"""
        counter = self.n
        dim = -1
        for i in range(self.n):
            if self.inf[i] == B.inf[i] and self.sup[i] == B.sup[i]:
                counter -= 1
            elif self.sup[i] == B.inf[i] or self.inf[i] == B.sup[i]:
                dim = i
        if counter == 1:
            return dim
        else:
            return -1

    def merge(self, B, dim):
        """Merges two interval vectors across an appropriate dimension"""
        C = ndInterval(self.n, [], [])
        for i in range(C.n):
            if i != dim:
                C.sup.append(self.sup[i])
                C.inf.append(self.inf[i])
            else:
                C.sup.append(self.sup[i] * (self.sup[i] >= B.sup[i]) + B.sup[i] * (self.sup[i] < B.sup[i]))
                C.inf.append(self.inf[i] * (self.inf[i] <= B.inf[i]) + B.inf[i] * (self.inf[i] > B.inf[i]))

        C.interval = [interval[C.inf[i], C.sup[i]] for i in range(C.n)]
        return C

    def search_merge(list):
        """Searches for intervals that can be merged together and merges them"""
        change = True
        n = len(list)
        while change and len(list) > 1:
            for A in list:
                for B in list:
                    dim = A.adjacency(B)
                    if dim > -1:
                        C = A.merge(B, dim)
                        change = True
                        list.remove(A)
                        list.remove(B)
                        list.append(C)
                        n = len(list)
                        break
                if A not in list:
                    continue
            if len(list) == n:
                change = False
        
        return list

    def split(self, dims, lows=[], ups=[], split_value=dict()):
        """Splits an interval across a dimension"""
        if not dims:
            return [self]
        if lows == [] or ups == []:
            lows = self.inf
            ups = self.sup
        if dims:
            d = dims[0]
            lows1 = copy.deepcopy(lows)
            ups1 = copy.deepcopy(ups)
            if d not in split_value.keys():
                ups1[d] = lows[d] + ((ups[d] - lows[d]) / 2)
            else:
                ups1[d] = split_value[d]
            partition1 = ndInterval(self.n, inf=lows1, sup=ups1)
            list1 = partition1.split(dims[1:], lows1, ups1)

            lows2 = copy.deepcopy(lows1)
            if d not in split_value.keys():
                lows2[d] = lows[d] + ((ups[d] - lows[d]) / 2)
            else:
                lows2[d] = split_value[d]
            ups2 = copy.deepcopy(ups)
            partition2 = ndInterval(self.n, inf=lows2, sup=ups2)
            list2 = partition2.split(dims[1:], lows2, ups2)

            return list1 + list2

    def complement(self, subinterval):
        """Computes the complement of a sub interval inside the original interval"""
        complement = []
        for v in range(self.n):
            inf1 = copy.copy(self.inf)
            sup1 = copy.copy(self.sup)
            sup1[v] = subinterval.inf[v]
            if sup1[v] > inf1[v]:
                int1 = ndInterval(self.n, inf=inf1, sup=sup1)
                complement.append(int1)

            inf2 = copy.copy(self.inf)
            inf2[v] = subinterval.sup[v]
            sup2 = copy.copy(self.sup)
            if sup2[v] > inf2[v]:
                int2 = ndInterval(self.n, inf=inf2, sup=sup2)
                complement.append(int2)

        ndInterval.search_merge(complement)
        return ndInterval.remove_duplicates(complement)

    def intersection(self, interval):
        intersection_inf = list(np.maximum(self.inf, interval.inf))
        intersection_sup = list(np.minimum(self.sup, interval.sup))

        # Empty intersection
        if max(np.array(intersection_inf) > np.array(intersection_sup)):
            return []
        else:
            return [intersection_inf, intersection_sup]

    def remove_duplicates(interval_list):
        """Takes a list of intervals and eliminates duplicate intersections"""
        for i in range(len(interval_list)):
            partition1 = interval_list[i]
            for j in range(i+1,len(interval_list)):
                partition2 = interval_list[j]
                intersection = partition1.intersection(partition2)
                if intersection:
                    new_inf = []
                    new_sup = []
                    for v in range(partition2.n):
                        if partition2.inf[v] < intersection[0][v]:
                            new_sup += [intersection[0][v]]
                        else:
                            new_sup += [partition2.sup[v]]
                        if partition2.sup[v] > intersection[1][v]:
                            new_inf += [intersection[1][v]]
                        else:
                            new_inf += [partition2.inf[v]]
                    interval_list[j] = ndInterval(partition2.n, new_inf, new_sup)


        return interval_list

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, maxsize=1e6):
        self.storage = [[] for _ in range(8)]
        self.maxsize = maxsize
        self.next_idx = 0

    def clear(self):
        self.storage = [[] for _ in range(8)]
        self.next_idx = 0

    # Expects tuples of (x, x', g, u, r, d, x_seq, a_seq)
    def add(self, data):
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            [array.append(datapoint) for array, datapoint in zip(self.storage, data)]
        else:
            [array.__setitem__(self.next_idx, datapoint) for array, datapoint in zip(self.storage, data)]

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        if len(self.storage[0]) <= batch_size:
            ind = np.arange(len(self.storage[0]))
        else:
            ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, y, g, u, r, d, x_seq, a_seq = [], [], [], [], [], [], [], []          

        for i in ind: 
            X, Y, G, U, R, D, obs_seq, acts = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

            # For off-policy goal correction
            x_seq.append(np.array(obs_seq, copy=False))
            a_seq.append(np.array(acts, copy=False))
        
        return np.array(x), np.array(y), np.array(g), \
            np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
            x_seq, a_seq

    def save(self, file):
        np.savez_compressed(file, idx=np.array([self.next_idx]), x=self.storage[0],
                            y=self.storage[1], g=self.storage[2], u=self.storage[3],
                            r=self.storage[4], d=self.storage[5], xseq=self.storage[6],
                            aseq=self.storage[7])

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data['idx'][0])
            self.storage = [data['x'], data['y'], data['g'], data['u'], data['r'],
                            data['d'], data['xseq'], data['aseq']]
            self.storage = [list(l) for l in self.storage]

    def __len__(self):
        return len(self.storage[0])


class PartitionBuffer:
    def __init__(self, maxsize=1e6):
        self.storage = [[] for _ in range(6)]
        self.maxsize = maxsize
        self.next_idx = 0

    def clear(self):
        self.storage = [[] for _ in range(6)]
        self.next_idx = 0

    # Expects tuples of (x, x', g, u, r, d, x_seq, a_seq)
    def add(self, data):
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            [array.append(datapoint) for array, datapoint in zip(self.storage, data)]
        else:
            [array.__setitem__(self.next_idx, datapoint) for array, datapoint in zip(self.storage, data)]

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        if len(self.storage[0]) <= batch_size:
            ind = np.arange(len(self.storage[0]))
        else:
            ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, gs, y, gt, rl, rh = [], [], [], [], [], []

        for i in ind: 
            X, Gs, Y, Gt, Rl, Rh = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            gs.append(np.array(Gs, copy=False))
            y.append(np.array(Y, copy=False))
            gt.append(np.array(Gt, copy=False))
            rl.append(np.array(Rl, copy=False))
            rh.append(np.array(Rh, copy=False))

        return np.array(x), np.array(gs), np.array(y), np.array(gt), \
            np.array(rl).reshape(-1, 1), np.array(rh).reshape(-1, 1)

    def target_sample(self, start_partition, target_partition, batch_size):
        indices = []
        for i in range(len(self.storage[0])):
            if (self.storage[1][i] == start_partition).all() and (self.storage[3][i] == target_partition).all():
                indices.append(i)

        if len(indices) == 0:
            if len(self.storage[0]) <= batch_size:
                ind = np.arange(len(self.storage[0]))
            else:
                ind = np.random.randint(0, len(self.storage[0]), size=batch_size)
        elif len(indices) <= batch_size:
            ind = indices
        else:            
            ind = np.random.choice(indices, size=batch_size)

        x, gs, y, gt, rl, rh = [], [], [], [], [], []

        for i in ind: 
            X, Gs, Y, Gt, Rl, Rh = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            gs.append(np.array(Gs, copy=False))
            y.append(np.array(Y, copy=False))
            gt.append(np.array(Gt, copy=False))
            rl.append(np.array(Rl, copy=False))
            rh.append(np.array(Rh, copy=False))

        return np.array(x), np.array(gs), np.array(y), np.array(gt), \
            np.array(rl).reshape(-1, 1), np.array(rh).reshape(-1, 1)

    def save(self, file):
        np.savez_compressed(file, idx=np.array([self.next_idx]), x=self.storage[0],
                            gs=self.storage[1], y=self.storage[2], gt=self.storage[3],
                            rl=self.storage[4], rh=self.storage[5])

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data['idx'][0])
            self.storage = [data['x'], data['gs'], data['y'], data['gt'], data['rl'],
                            data['rh']]
            self.storage = [list(l) for l in self.storage]

    def __len__(self):
        return len(self.storage[0])


class TrajectoryBuffer(object):

    def __init__(self, capacity):
        self._capacity = capacity
        self.reset()

    def reset(self):
        self._num_traj = 0  # number of trajectories
        self._size = 0    # number of game frames
        self.trajectory = []

    def __len__(self):
        return self._num_traj

    def size(self):
        return self._size

    def get_traj_num(self):
        return self._num_traj

    def full(self):
        return self._size >= self._capacity

    def create_new_trajectory(self):
        self.trajectory.append([])
        self._num_traj += 1

    def append(self, s):
        self.trajectory[self._num_traj-1].append(s)
        self._size += 1

    def get_trajectory(self):
        return self.trajectory

    def set_capacity(self, new_capacity):
        assert self._size <= new_capacity
        self._capacity = new_capacity


class NormalNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        action = (action + np.random.normal(0, self.sigma,
            size=action.shape)).clip(min_action, max_action)
        return action


class OUNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return (self.X + action).clip(min_action, max_action)


def train_forward_model(forward_model, partition_buffer, Gs=None, Gt=None, n_epochs=100, batch_size=64, device='cpu', verbose=False):
    if Gs is not None and Gt is not None:
        x, gs, y, gt, rl, rh = partition_buffer.target_sample(Gs, Gt, batch_size)
    else:        
        x, gs, y, gt, rl, rh = partition_buffer.sample(batch_size)
        
    forward_model.fit(x, gt, y, n_epochs=n_epochs, verbose=verbose)

def train_adj_net(a_net, states, adj_mat, optimizer, margin_pos, margin_neg,
                  n_epochs=100, batch_size=64, device='cpu', verbose=False):
    if verbose:
        print('Generating training data...')
    dataset = MetricDataset(states, adj_mat)
    if verbose:
        print('Totally {} training pairs.'.format(len(dataset)))
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    n_batches = len(dataloader)

    loss_func = ContrastiveLoss(margin_pos, margin_neg)

    for i in range(n_epochs):
        epoch_loss = []
        for j, data in enumerate(dataloader):
            x, y, label = data
            x = x.float().to(device)
            y = y.float().to(device)
            label = label.long().to(device)
            x = a_net(x)
            y = a_net(y)
            loss = loss_func(x, y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and (j % 50 == 0 or j == n_batches - 1):
                print('Training metric network: epoch {}/{}, batch {}/{}'.format(i+1, n_epochs, j+1, n_batches))

            epoch_loss.append(loss.item())

        if verbose:
            print('Mean loss: {:.4f}'.format(np.mean(epoch_loss)))


class ContrastiveLoss(nn.Module):

    def __init__(self, margin_pos, margin_neg):
        super().__init__()
        assert margin_pos <= margin_neg
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(self, x, y, label):
        # mutually reachable states correspond to label = 1
        dist = torch.sqrt(torch.pow(x - y, 2).sum(dim=1) + 1e-12)
        loss = (label * (dist - self.margin_pos).clamp(min=0)).mean() + ((1 - label) * (self.margin_neg - dist).clamp(min=0)).mean()
        return loss


class MetricDataset(Data.Dataset):

    def __init__(self, states, adj_mat):
        super().__init__()
        n_samples = adj_mat.shape[0]
        self.x = []
        self.y = []
        self.label = []
        for i in range(n_samples - 1):
            for j in range(i + 1, n_samples):
                self.x.append(states[i])
                self.y.append(states[j])
                self.label.append(adj_mat[i, j])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.label = np.array(self.label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.label[idx]


class PartitionDataset(Data.Dataset):

    def __init__(self, partition_buffer, batch_size):
        super().__init__()
        n_samples = len(partition_buffer.storage[0])
        self.state = []
        self.target_partition = []
        self.reached_state = []
        x, gs, y, gt, rl, rh = partition_buffer.sample(batch_size)

        self.state = x
        self.target_partition = gt
        self.reached_state = y

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        return self.state[idx], self.target_partition[idx], self.reached_state[idx]
    

def manager_mapping(grid, g_low, g_high, file, resolution=100):
    """ plots a heatmap of the manager's subgoals and save it to a file """
    ax = sns.heatmap(grid,cmap="viridis", cbar=False)
    ax.invert_yaxis()
    plt.savefig(file)
    plt.close()