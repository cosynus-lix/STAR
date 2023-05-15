import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from keras.layers import Input, Dense
from keras.optimizers import Adam

from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super().__init__()

        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
    
    def forward(self, x, g=None, nonlinearity='tanh'):
        if g is not None:
            x = F.relu(self.l1(torch.cat([x, g], 1)))
        else:
            x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if nonlinearity == 'tanh':
            x = self.max_action * torch.tanh(self.l3(x)) 
        elif nonlinearity == 'sigmoid':
            x = self.max_action * torch.sigmoid(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, g=None, u=None):
        if g is not None:
            xu = torch.cat([x, g, u], 1)
        else:
            xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, g=None, u=None):
        if g is not None:
            xu = torch.cat([x, g, u], 1)
        else:
            xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class ControllerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=1):
        super().__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        self.scale = nn.Parameter(torch.tensor(scale).float(),
                                  requires_grad=False)
        self.actor = Actor(state_dim, goal_dim, action_dim, 1)
    
    def forward(self, x, g):
        return self.scale*self.actor(x, g)


class ControllerCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()

        self.critic = Critic(state_dim, goal_dim, action_dim)
    
    def forward(self, x, sg, u):
        return self.critic(x, sg, u)

    def Q1(self, x, sg, u):
        return self.critic.Q1(x, sg, u)


class ManagerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=None, absolute_goal=False):
        super().__init__()
        if scale is None:
            scale = torch.ones(action_dim)
        self.scale = nn.Parameter(torch.tensor(scale[:action_dim]).float(), requires_grad=False)
        self.actor = Actor(state_dim, goal_dim, action_dim, 1)
        self.absolute_goal = absolute_goal
    
    def forward(self, x, g):
        if self.absolute_goal:
            return self.scale * self.actor(x, g, nonlinearity='sigmoid')
        else:
            return self.scale * self.actor(x, g)


class ManagerCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.critic = Critic(state_dim, goal_dim, action_dim)

    def forward(self, x, g, u):
        return self.critic(x, g, u)

    def Q1(self, x, g, u):
        return self.critic.Q1(x, g, u)


class ANet(nn.Module):

    def __init__(self, state_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# class ForwardModel(nn.Module):

#     def __init__(self, state_dim, goal_dim, hidden_dim):
#         super().__init__()
#         self.state_dim = state_dim
#         self.goal_dim = goal_dim
#         self.fc1 = nn.Linear(state_dim + goal_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, state_dim)
    
#     def forward(self, s, g):
#         x = F.relu(self.fc1(torch.cat([s, g], 1)))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
    
class ForwardModel():

    def __init__(self, state_dim, goal_dim, hidden_dim, learning_rate):
        self.goal_dim = goal_dim
        self.state_dim = goal_dim // 2
        self.model = tf.keras.Sequential([
            Input((self.state_dim + self.goal_dim,)),
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu'),
            Dense(self.state_dim)
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate))
    
    def fit(self, states, goals, reached_states, n_epochs=100, verbose=False):
        input = tf.concat((states[:, :self.state_dim], goals), axis=1)
        self.model.fit(input, reached_states[:, :self.state_dim], epochs=n_epochs, verbose=verbose)
    
    def predict(self, states, goals, verbose=False):
        input = tf.concat((states[:, :self.state_dim], goals), axis=1)
        return self.model.predict(input, verbose=verbose)
    
    def measure_error(self, partition_buffer, batch_size):
        x, gs, y, gt, rl, rh = partition_buffer.sample(batch_size)
        loss = mean_squared_error(y[:, :self.state_dim], self.predict(x, gs))
        return loss

    def save(self, path):
        self.model.save(path)

    
# def convert_h5(torch_model):
#     """convert model to h5"""
#     torch_model.eval()
#     dummy_state = torch.randn(1, torch_model.state_dim).to(device)
#     dummy_goal = torch.randn(1, torch_model.goal_dim).to(device)
#     torch.onnx.export(torch_model,               
#         (dummy_state, dummy_goal),
#         "forward_model.onnx",   
#         export_params=True,        
#         opset_version=10,          
#         do_constant_folding=True,  
#         input_names = ['state', 'goal'],   
#         output_names = ['output']
#         )
#     onnx_model = onnx.load("forward_model.onnx")
#     onnx.checker.check_model(onnx_model)
#     k_model = onnx_to_keras(onnx_model, ['state', 'goal'])
    # tf_rep = prepare(onnx_model)
    # tf_rep.export_graph("forward_model.h5")
