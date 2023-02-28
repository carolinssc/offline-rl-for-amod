"""
Contains a heuristic for a greedy rebalancing policy
"""
import numpy as np


class Heuristic():
    def __init__(self, p=0, horizon=6):
        self.p = p
        self.horizon = horizon
        super().__init__()

    def next_action(self, env):
        demand = [[sum([(env.scenario.demand_input[i, j][t]) for j in env.region])
                   for i in env.region] for t in range(env.time+1, env.time+self.horizon)]
        d = np.array(demand)
        d = list(d.sum(axis=0)/d.sum())
        return d
