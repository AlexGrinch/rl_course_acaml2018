import random
import numpy as np
from collections import namedtuple


class ReplayBuffer:

    def __init__(self, capacity):
        self.size = capacity
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.idx = 0
        self.stored_in_buffer = 0
        self.transition = namedtuple(
            "Transition", ("s", "a", "r", "s_", "done"))
        
    def push_transition(self, transition):
        """ transition = [state, action, reward, next_state, done]
        """
        state, action, reward, next_state, done = transition

        if self.states is None:
            self.states = np.empty(
                [self.size] + list(state.shape), dtype=np.uint8)
            self.actions = np.empty(
                [self.size], dtype=np.int32)
            self.rewards = np.empty(
                [self.size], dtype=np.float32)
            self.next_states = np.empty(
                [self.size] + list(state.shape), dtype=np.uint8)
            self.dones = np.empty(
                [self.size], dtype = np.bool)

        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.size
        self.stored_in_buffer = min(self.size, self.stored_in_buffer + 1)
        
    def get_batch(self, batch_size):
        indices = random.sample(
            range(self.stored_in_buffer), k=batch_size)
        s = self.states[indices]
        a = self.actions[indices]
        r = self.rewards[indices]
        s_ = self.next_states[indices]
        done = self.dones[indices]
        batch = self.transition(s, a, r, s_, done)
        return batch
