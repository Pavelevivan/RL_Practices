import gym
import torch
from torch import nn
import numpy as np
#lunar lander
# acrobat
# mountain_car
class Agent(nn.Module):
    def __init__(self, states_dim, actions_dim, epsilon):
        self.linear_1 = nn.Linear(states_dim, 50)
        self.linear_2 = nn.Linear(50, 20)
        self.linear_3 = nn.Linear(20, actions_dim)
        self.relu = nn.ReLU()
        self.epsilon = epsilon
        self.actions_dim = actions_dim
        self.states_dim = states_dim
        loss =nn.MSELoss()
    
    def forward(self, state):
        hidden = self.linear_1(state)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        hidden = self.linear_3(hidden)
        return nn.Softmax(hidden)

    def get_action(self, state):
        action_prob = (1-self.epsilon)*self.forward(state) + self.epsilon/self.actions_dim
        action = np.random.choice(list(range(self.actions_dim), p=action_prob))
        return action

    def update_policy(self, elite_sessions):
        elite_states, elite_actions = map(torch.tensor, *zip(elite_sessions))
        loss = nn.
