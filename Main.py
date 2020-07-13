from EnvHelper import EnvHelper
from Buffer import Buffer
from Agent import Agent
from OUP import Noise
from NeuralNetworks import enable_cuda


if __name__ == '__main__':
    enable_cuda()
    env_helper = EnvHelper()
    env, n_states, n_actions = env_helper.make_environment('LunarLanderContinuous-v2')
    # env_helper.env.action_space.seed(101010)
    # env_helper.env.seed(101010)

    buffer = Buffer(capacity=1000000, batch_size=128)
    noise_object = Noise(n_actions, mu=0, sigma=0.2)
    agent = Agent(
        buffer=buffer,
        noise_object=noise_object,
        environment_helper=env_helper,
        name='Lander',
        learning_factor=0.001,
        checkpoint_frequency=50,
        critic_lr=0.001
    )

    agent.run(max_iterations=2000, train=True, render=True, verbose=True, early_termination=1000000000000)
    # agent.load(700)
    agent.run(10, train=False, render=True, verbose=True, checkpoint=False)