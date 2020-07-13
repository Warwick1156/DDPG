import gym


class EnvHelper:
    def __init__(self):
        self.n_states = None
        self.n_actions = None
        self.action_bounds = []
        self.env = None
        self.action_upper_limit = None

    def make_environment(self, env_name):
        try:
            print('LOADING {} ENVIRONMENT'.format(env_name))
            self.env = gym.make(env_name)
        except:
            raise Exception('Error occurred during environment making. Check environment name.')

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.action_upper_limit = self.env.action_space.high[0]

        print('Input features: {}'.format(self.n_states))
        print('Output features: {}'.format(self.n_actions))

        for i in range(self.n_actions):
            lower_bound = self.env.action_space.low[i]
            upper_bound = self.env.action_space.high[i]

            print('Action {0} lower bound: {1}'.format(i, lower_bound))
            print('Action {0} upper bound: {1}'.format(i, upper_bound))
            self.action_bounds.append((lower_bound, upper_bound))

        return self.env, self.n_states, self.n_actions
