import numpy as np


class TrajBatchDisc:

    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.states = list(next(self.batch))
        self.actions = list(next(self.batch))
        self.next_terminations = np.stack(next(self.batch))
        self.next_dones = np.stack(next(self.batch))
        self.next_states = list(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))
        try:
            self.agent_ids = np.stack(next(self.batch)).astype(np.int64)
        except StopIteration:
            self.agent_ids = np.zeros_like(self.rewards, dtype=np.int64)


def init_fc_weights(fc):
    fc.weight.data.mul_(0.1)
    fc.bias.data.mul_(0.0)