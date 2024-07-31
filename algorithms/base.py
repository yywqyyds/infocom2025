from abc import ABCMeta, abstractmethod
import numpy as np
from arms import StrategicArm  # 假设 StrategicArm 类定义在 arm.py 文件中


class BaseAlgorithm(metaclass=ABCMeta):
    def __init__(self, tasks: list, workers: list, num_tasks: int, num_workers: int, budget: float):
        super().__init__()
        self.tasks = tasks  # 任务集合
        self.workers = workers  # 工人集合
        self.N = num_tasks  # 任务总数
        self.W = num_workers  # 工人总数
        self.B = budget  # 预算
        self.t = 0  # 当前轮次索引
        self.R = 0.0  # 总奖励

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def omniscience(self):
        # 返回所有工人对所有任务的预期奖励
        rewards = np.zeros((self.W, self.N))
        for i, worker in enumerate(self.workers):
            for j, task in enumerate(self.tasks):
                arm = StrategicArm()
                rewards[i, j] = arm.draw()  # 假设 draw() 方法返回一个预期奖励
        return rewards


