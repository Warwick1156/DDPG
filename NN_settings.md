# BipedalWalker-v3

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

learning_factor=0.001
mu=0