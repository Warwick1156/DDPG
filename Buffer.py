from collections import deque
from torch.autograd import Variable
from torch import from_numpy

import numpy as np
import random

STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3


class Buffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.counter = 0

        self.batch_size = batch_size

    def __call__(self):
        assert_len = min(self.batch_size, self.counter)
        sample_buffer = random.sample(self.buffer, assert_len)

        state = Variable(from_numpy(np.float32([x[STATE] for x in sample_buffer])))
        action = Variable(from_numpy(np.float32([x[ACTION] for x in sample_buffer])))
        reward = Variable(from_numpy(np.float32([x[REWARD] for x in sample_buffer])))
        next_state = Variable(from_numpy(np.float32([x[NEXT_STATE] for x in sample_buffer])))

        return state, action, reward, next_state

    def record(self, state, action, reward, next_state):
        if self.counter < self.capacity:
            self.counter += 1
        self.buffer.append((state, action, reward, next_state))