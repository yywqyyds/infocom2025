import random
from algorithms.base import BaseAlgorithm
from arms import generate_tasks, generate_workers, StrategicArm

class RandomAlgorithm(BaseAlgorithm):
    def __init__(self, tasks, workers, num_tasks, num_workers, budget, c_max, alpha, beta):
        super().__init__(tasks, workers, num_tasks, num_workers, budget)
        self.c_max = c_max  # 最大可能成本
        self.alpha = alpha  # 超参数 alpha
        self.beta = beta  # 超参数 beta
        self.t = 0
        self.R = 0

    def initialize(self):
        # 初始化方法，不需要特殊处理
        pass

    def run(self):
        total_reward = 0
        remaining_budget = self.B
        round_counter = 0

        while remaining_budget > 0:
            # 随机选择任务
            task = random.choice(self.tasks)
            required_workers = task['required_workers']
            # 随机选择工人
            selected_workers = random.sample(self.workers, required_workers)

            total_cost =required_workers * self.c_max
            if total_cost > remaining_budget:
                break

            remaining_budget -= total_cost
            total_quality = sum(StrategicArm(mu=worker[f"task_{task['id']}_expected_quality"], sigma=worker[f"task_{task['id']}_sigma"]).draw()
                                for worker in selected_workers)

            total_reward += total_quality
            round_counter += 1
            self.t += 1

        return total_reward, round_counter


if __name__ == "__main__":
    num_tasks = 2
    num_workers = 3
    budget = 40
    c_max = 1
    alpha = 1 / 8
    beta = 1 / 8
    tasks = generate_tasks(num_tasks)
    workers = generate_workers(num_workers, tasks)

    algo = RandomAlgorithm(tasks, workers, num_tasks, num_workers, budget, c_max, alpha, beta)
    total_reward, round_counter = algo.run()
    print(total_reward, round_counter)