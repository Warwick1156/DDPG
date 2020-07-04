from EnvHelper import EnvHelper
from Buffer import Buffer
from Agent import Agent
from OUP import Noise
from NeuralNetworks import enable_cuda


if __name__ == '__main__':
    enable_cuda()
    env_helper = EnvHelper()
    env, n_states, n_actions = env_helper.make_environment('BipedalWalker-v3')

    buffer = Buffer(capacity=1000000, batch_size=256)
    noise_object = Noise(n_actions)
    agent = Agent(
        buffer=buffer,
        noise_object=noise_object,
        environment_helper=env_helper,
        name='Walker',
        learning_factor=0.001,
    )

    agent.run(max_iterations=2000, train=True, render=True, verbose=True, early_termination=1000)
    # agent.load(900)
    # agent.run(10, train=False, render=True, verbose=True, checkpoint=False)