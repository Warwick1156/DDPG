from NeuralNetworks import Actor, Critic, get_optimizer, calculate_loss, to_tensor
from torch import squeeze, sum, save, load

import gc
import numpy as np

DIR = './Models/'
ACTOR = '_ACTOR'
CRITIC = '_CRITIC'
EXTENSION = '.pt'


class Agent:
    def __init__(self, buffer, noise_object, environment_helper, actor_lr=0.001, critic_lr=0.001, name='Default_Agent_Name', set_up_nn=True, learning_factor=0.005, gamma=0.99, checkpoint_frequency=100):
        self.buffer = buffer
        self.noise = noise_object
        self.env = environment_helper

        self.learning_factor = learning_factor
        self.gamma = gamma
        self.name = name
        self.checkpoint_frequency = checkpoint_frequency

        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None

        self.actor_optimizer = None
        self.critic_optimizer = None

        if set_up_nn:
            self.set_up_neural_networks(actor_lr, critic_lr)

    def set_up_neural_networks(self, actor_lr, critic_lr):
        # All action limits are the same,
        # so I took limit from first action as env.action_bounds[0] and [1] because upper limit
        self.actor = Actor(self.env.n_states, self.env.n_actions, self.env.action_bounds[0][1])
        self.actor_target = Actor(self.env.n_states, self.env.n_actions, self.env.action_bounds[0][1])
        self.actor_optimizer = get_optimizer(target=self.actor.parameters(), learning_rate=actor_lr)

        self.critic = Critic(self.env.n_states, self.env.n_actions)
        self.critic_target = Critic(self.env.n_states, self.env.n_actions)
        self.critic_optimizer = get_optimizer(target=self.critic.parameters(), learning_rate=critic_lr)

        self.equalize_weights()

    def equalize_weights(self):
        self.load_weights(self.actor, self.actor_target)
        self.load_weights(self.critic, self.critic_target)

    def load_weights(self, source, target):
        for source_weight, target_weight in zip(source.parameters(), target.parameters()):
            target_weight.data.copy_(source_weight.data)

    def update_weights(self, source, target):
        for source_weight, target_weight in zip(source.parameters(), target.parameters()):
            target_weight.data.copy_(target_weight.data * (1. - self.learning_factor) + source_weight * self.learning_factor)

    def update_gradient(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def learn(self):
        state, action, reward, next_state = self.buffer()
        self.learn_critic(state, action, reward, next_state)
        self.learn_actor(state, action, reward, next_state)

        self.update_weights(self.actor, self.actor_target)
        self.update_weights(self.critic, self.critic_target)

    def learn_actor(self, state, action, reward, next_state):
        predicted_action = self.actor.forward(state)
        actor_loss = - sum(self.critic.forward(state, predicted_action))
        self.update_gradient(self.actor_optimizer, actor_loss)

    def learn_critic(self, state, action, reward, next_state):
        next_action = self.actor_target.forward(next_state).detach()
        q_value = squeeze(self.critic_target.forward(next_state, next_action))
        expected_reward = reward + self.gamma * q_value
        predicted_reward = squeeze(self.critic.forward(state, action))
        critic_loss = calculate_loss(predicted_reward, expected_reward)
        self.update_gradient(self.critic_optimizer, critic_loss)

    def run(self, max_iterations, train, render, verbose, early_termination=25000, checkpoint=True):
        for iteration in range(max_iterations):
            episode_reward = self.run_episode(train, render, early_termination)
            if verbose:
                print('Iteration {0} score: {1}'.format(iteration, episode_reward))
            gc.collect()
            if checkpoint:
                self.checkpoint(iteration)

    def run_episode(self, train, render, early_termination):
        initial_state = self.env.env.reset()
        episode_reward = 0
        counter = 0
        while True:
            if render:
                self.env.env.render()

            state = np.float32(initial_state)
            action = self.policy(state, train)
            new_state, reward, done, _ = self.env.env.step(action)
            episode_reward += reward
            next_state = np.float32(new_state)

            if train and not done:
                self.buffer.record(state, action, reward, next_state)
                self.learn()
            initial_state = new_state

            if done or self.terminate(counter, early_termination):
                break
            counter += 1

        return episode_reward

    def terminate(self, i, stop):
        if i >= stop:
            return True
        else:
            return False

    def policy(self, state, train):
        state_tensor = to_tensor(state)
        if train:
            action = self.exploration(state_tensor)
        else:
            action = self.exploitation(state_tensor)
        return action

    # TODO: Change env.action_bounds[0][1] to make it look more obvious
    def exploration(self, state):
        action = self.actor.forward(state).detach()
        noise = self.noise()
        return action.cpu().data.numpy() + (noise * self.env.action_bounds[0][1])

    def exploitation(self, state):
        action = self.actor_target.forward(state).detach()
        return action.cpu().data.numpy()

    def save(self, episode):
        save(self.actor_target.state_dict(), DIR + self.name + str(episode) + ACTOR + EXTENSION)
        save(self.critic_target.state_dict(), DIR + self.name + str(episode) + CRITIC + EXTENSION)
        print('Models saved')

    def load(self, episode):
        self.actor.load_state_dict(load(DIR + self.name + str(episode) + ACTOR + EXTENSION))
        # self.critic.load_state_dict(load(DIR + self.name + str(episode) + CRITIC + EXTENSION))
        self.load_weights(self.actor, self.actor_target)
        # self.load_weights(self.critic, self.critic_target)

    def checkpoint(self, episode):
        try:
            if episode % self.checkpoint_frequency == 0:
                self.save(episode)
        except:
            pass
