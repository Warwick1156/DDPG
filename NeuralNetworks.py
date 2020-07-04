import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional
from torch.optim import Adam
from torch.autograd import Variable


def enable_cuda():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.tensor(1, dtype=torch.float, requires_grad=True).cuda()


def get_optimizer(target, learning_rate):
    return Adam(target, learning_rate)


def calculate_loss(predicted, expected):
    return functional.smooth_l1_loss(predicted, expected)


def to_tensor(data):
    return Variable(torch.from_numpy(data))


class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.states_input = nn.Linear(n_states, 256)
        self.states_input.weight.data = uniform_weights(self.states_input.weight.data.size())
        self.states_hidden = nn.Linear(256, 128)
        self.states_hidden.weight.data = uniform_weights(self.states_hidden.weight.data.size())

        self.action_input = nn.Linear(n_actions, 128)
        self.action_input.weight.data = uniform_weights(self.action_input.weight.data.size())

        self.concatenated = nn.Linear(256, 128)
        self.concatenated.weight.data = uniform_weights(self.concatenated.weight.data.size())

        self.output = nn.Linear(128, 1)
        self.output.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        x_input = functional.relu(self.states_input(state.cuda()))
        x_critic_state = functional.relu(self.states_hidden(x_input))
        critic_action = functional.relu(self.action_input(action.cuda()))
        concatenate = torch.cat((x_critic_state, critic_action), dim=1)
        concatenate = functional.relu(self.concatenated(concatenate))
        return self.output(concatenate)


class Actor(nn.Module):
    def __init__(self, n_state, n_action, limit):
        super(Actor, self).__init__()

        self.n_state = n_state
        self.n_action = n_action
        self.limit = limit

        self.input = nn.Linear(n_state, 256)
        self.input.weight.data = uniform_weights(self.input.weight.data.size())

        self.hidden_1 = nn.Linear(256, 128)
        self.hidden_1.weight.data = uniform_weights(self.hidden_1.weight.data.size())

        self.hidden_2 = nn.Linear(128, 64)
        self.hidden_2.weight.data = uniform_weights(self.hidden_2.weight.data.size())

        self.output = nn.Linear(64, n_action)
        self.output.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        x = functional.relu(self.input(state.cuda()))
        x = functional.relu(self.hidden_1(x))
        x = functional.relu(self.hidden_2(x))
        return torch.tanh(self.output(x)) * self.limit


def uniform_weights(size, distribution=None):
    distribution = distribution or size[0]
    value = 1.0 / np.sqrt(distribution)
    return torch.Tensor(size).uniform_(-value, value)
