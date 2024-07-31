import matplotlib.pyplot as plt
import numpy as np
from algorithms.algorithms1 import ExplorationPriorToExploitation
from algorithms.algorithms2 import ExplorationAndExploitation
from algorithms.opt import Opt
from algorithms.random import RandomAlgorithm
from arms import generate_tasks, generate_workers, StrategicArm

class Emulator:
    algorithms = ['ExplorationPriorToExploitation', 'ExplorationAndExploitation', 'Opt', 'RandomAlgorithm']

    def __init__(self, tasks, workers, num_tasks, num_workers, budget, c_max, alpha, beta):
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.budget = budget
        self.c_max = c_max
        self.alpha = alpha
        self.beta = beta

        self.tasks = tasks
        self.workers = workers

        self.name2sol = {}

    def build(self):
        for algo in Emulator.algorithms:
            if algo == 'ExplorationPriorToExploitation':
                self.name2sol[algo] = ExplorationPriorToExploitation(self.tasks, self.workers, self.num_tasks, self.num_workers, self.budget, self.c_max, self.alpha, self.beta)
            elif algo == 'ExplorationAndExploitation':
                self.name2sol[algo] = ExplorationAndExploitation(self.tasks, self.workers, self.num_tasks, self.num_workers, self.budget, self.c_max, self.alpha, self.beta)
            elif algo == 'Opt':
                self.name2sol[algo] = Opt(self.tasks, self.workers, self.num_tasks, self.num_workers, self.budget)

            elif algo == 'RandomAlgorithm':
                self.name2sol[algo] = RandomAlgorithm(self.tasks, self.workers, self.num_tasks, self.num_workers, self.budget,
                                                      self.c_max, self.alpha, self.beta)
    def simulate(self):
        self.build()
        name2res = {name: None for name in self.name2sol.keys()}
        for name in name2res.keys():
            # instance of an algorithm
            solver = self.name2sol[name]
            solver.initialize()
            name2res[name] = solver.run()
        return name2res

