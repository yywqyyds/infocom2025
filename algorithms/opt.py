import numpy as np
from algorithms.base import BaseAlgorithm
from arms import generate_tasks, generate_workers, StrategicArm

class Opt(BaseAlgorithm):
    def __init__(self, tasks: list, workers: list, n_tasks: int, n_workers: int, budget: float):
        super().__init__(tasks, workers, n_tasks, n_workers, budget)

    def initialize(self):
        """ 对每个任务，根据工人的 p.p.r (quality / bid) 值对工人进行排序 """
        for task in self.tasks:
            # 对每个任务，根据工人的性价比 (quality / bid) 值对工人进行排序
            task['workers'] = sorted(
                self.workers,
                key=lambda worker: worker[f"task_{task['id']}_expected_quality"] / worker[f"task_{task['id']}_bid"],
                reverse=True
            )

    def run(self):
        """ 每轮选择一个性价比最高的任务，并为该任务选择性价比最高的工人 """
        total_reward = 0
        round_counter = 0

        while self.B > 0:
            # 计算每个任务的性价比 (TCER_j)
            def compute_tcer(task):
                total_bid = sum(worker[f"task_{task['id']}_bid"] for worker in task['workers'][:task['required_workers']])
                total_quality = sum(worker[f"task_{task['id']}_expected_quality"] for worker in task['workers'][:task['required_workers']])
                tcer = total_quality / total_bid
                return tcer

            # 对任务按照性价比 (TCER_j) 排序
            self.tasks.sort(key=compute_tcer, reverse=True)

            # 选择性价比最高的任务
            best_task = self.tasks[0]
            required_workers = best_task['required_workers']
            selected_workers = best_task['workers'][:required_workers]

            # 计算选择的工人的总支付和实际质量
            total_bid = sum(worker[f"task_{best_task['id']}_bid"] for worker in selected_workers)
            total_quality = sum(StrategicArm(mu=worker[f"task_{best_task['id']}_expected_quality"], sigma=worker[f"task_{best_task['id']}_sigma"]).draw() for worker in selected_workers)

            # 如果总支付超过剩余预算，停止运行
            if total_bid > self.B:
                break

            # 更新总奖励和预算
            total_reward += total_quality
            self.B -= total_bid

            # 增加轮次
            round_counter += 1
            self.t += 1

        return total_reward, round_counter

if __name__ == "__main__":
    num_tasks = 2
    num_workers = 3
    tasks = generate_tasks(num_tasks)
    workers = generate_workers(num_workers, tasks)

    algo = Opt(tasks, workers, n_tasks=num_tasks, n_workers=num_workers, budget=40)
    algo.initialize()
    reward, rounds = algo.run()

    print(f"Total reward: {reward}, Rounds: {rounds}")
