import numpy as np
from algorithms.base import BaseAlgorithm
from arms import generate_tasks, generate_workers, StrategicArm

class ExplorationAndExploitation(BaseAlgorithm):
    def __init__(self, tasks, workers, num_tasks, num_workers, budget, c_max, alpha, beta):
        super().__init__(tasks, workers, num_tasks, num_workers, budget)
        self.c_max = c_max  # 最大可能成本
        self.alpha = alpha  # 超参数 alpha
        self.beta = beta  # 超参数 beta
        self.t = 0
        self.R = 0
        self.n_j = {task["id"]: 0 for task in self.tasks}  # 任务 j 被选中的次数
        self.q_hat_j = {task["id"]: 0 for task in self.tasks}  # 任务 j 的平均奖励
        self.n_ij = {worker["id"]: {task["id"]: 0 for task in self.tasks} for worker in self.workers}  # 工人 i 执行任务 j 的次数
        self.q_hat_ij = {worker["id"]: {task["id"]: 0 for task in self.tasks} for worker in self.workers}  # 工人 i 在任务 j 上的平均质量
        self.q_hat_j_plus = {task["id"]: 0 for task in self.tasks}  # 任务 j 的 UCB 指数
        self.q_hat_ij_plus = {worker["id"]: {task["id"]: 0 for task in self.tasks} for worker in self.workers}  # 工人 i 在任务 j 上的 UCB 指数

        # for plotting individual rationality (real cost v.s. payment)
        self.cp1, self.cp2 = list(), list()


    def update_parameters(self, task, selected_workers, rewards, qualities):
        # 更新探索和利用阶段的数据
        self.n_j[task["id"]] += 1  # 更新任务 j 被选中的次数
        # 更新任务 j 的平均奖励
        self.q_hat_j[task["id"]] = (self.q_hat_j[task["id"]] * (self.n_j[task["id"]] - 1) + rewards) / self.n_j[task["id"]]
        # 更新任务 j 的 UCB 指数
        self.q_hat_j_plus[task["id"]] = self.q_hat_j[task["id"]] + np.sqrt(self.alpha * np.log(self.t + 1) / self.n_j[task["id"]])

        for worker, quality in zip(selected_workers, qualities):
            # 更新工人 i 执行任务 j 的次数
            self.n_ij[worker["id"]][task["id"]] += 1
            # 更新工人 i 在任务 j 上的平均质量
            self.q_hat_ij[worker["id"]][task["id"]] = (self.q_hat_ij[worker["id"]][task["id"]] * (self.n_ij[worker["id"]][task["id"]] - 1) + quality) / self.n_ij[worker["id"]][task["id"]]
            # 更新工人 i 在任务 j 上的 UCB 指数
            self.q_hat_ij_plus[worker["id"]][task["id"]] = self.q_hat_ij[worker["id"]][task["id"]] + np.sqrt(self.alpha * np.log(sum(self.n_ij[worker["id"]].values())) / self.n_ij[worker["id"]][task["id"]])

    def exploration_phase(self, verbose=False):
        # 探索阶段
        for task in self.tasks:
            worker_index = 0  # 初始化工人索引
            while worker_index < len(self.workers):
                selected_workers = []  # 初始化选中的工人列表
                for _ in range(task["required_workers"]):
                    selected_worker = self.workers[worker_index % len(self.workers)]  # 循环选择工人
                    selected_workers.append(selected_worker)
                    worker_index += 1

                total_cost = self.c_max * len(selected_workers)  # 计算总成本
                if verbose:
                    for w in selected_workers:
                        self.cp1.append((w[f"task_{task['id']}_bid"], self.c_max))
                if total_cost > self.B:
                    break  # 如果总成本超过剩余预算，停止探索

                self.t += 1  # 增加轮次
                # 随机生成工人的实际质量
                qualities = [StrategicArm(mu=worker[f"task_{task['id']}_expected_quality"],sigma=worker[f"task_{task['id']}_sigma"]).draw() for worker in selected_workers]
                rewards = sum(qualities)  # 计算总奖励
                self.R += rewards
                self.update_parameters(task, selected_workers, rewards, qualities)  # 更新探索数据
                self.B -= total_cost  # 更新已花费的预算


    def exploitation_phase(self, verbose=False):
        # 利用阶段
        while self.B > 0:
            def compute_tcer(task):
                # 计算任务的性价比 (TCER)
                total_bid = sum(worker[f"task_{task['id']}_bid"] for worker in self.workers)  # 计算总出价
                tcer = self.q_hat_j_plus[task["id"]] / (task['required_workers'] / len(self.workers) * total_bid)  # 计算任务的性价比
                return tcer

            self.tasks.sort(key=compute_tcer, reverse=True)  # 按性价比排序任务

            best_task = self.tasks[0]  # 选择性价比最高的任务
            required_workers = best_task['required_workers']

            def compute_wcer(worker):
                # 计算工人的性价比 (WCER)
                return self.q_hat_ij_plus[worker["id"]][best_task["id"]] / worker[f"task_{best_task['id']}_bid"]

            # 对该任务的工人按性价比排序
            sorted_workers = sorted(self.workers, key=compute_wcer, reverse=True)
            selected_workers = sorted_workers[:required_workers]  # 选择任务所需的工人
            # 计算支付并检查剩余预算是否足够
            total_payment = 0
            payments = []
            for worker in selected_workers:
                # 支付的时候要和临界工人比，除以临界工人的UCB值再乘以临界工人的报价 2024.07.26
                p_ij = min(self.q_hat_ij_plus[worker["id"]][best_task["id"]] / self.q_hat_ij_plus[sorted_workers[required_workers]["id"]][best_task["id"]] * sorted_workers[required_workers][f"task_{best_task['id']}_bid"], self.c_max)
                if verbose:
                    self.cp2.append((worker[f"task_{best_task['id']}_bid"], p_ij))
                total_payment += p_ij
                payments.append(p_ij)

            if total_payment > self.B:
                break  # 如果总支付超过剩余预算，停止利用阶段

            # 更新剩余预算
            self.B -= total_payment

            # 工人们对任务的完成质量在预期质量附近随机生成，并加到总奖励上
            total_quality = 0
            for worker, payment in zip(selected_workers, payments):
                actual_quality = StrategicArm(mu=worker[f"task_{best_task['id']}_expected_quality"], sigma=worker[f"task_{best_task['id']}_sigma"]).draw()
                total_quality += actual_quality


            self.R += total_quality  # 更新总奖励
            self.t += 1  # 增加轮次


            self.update_parameters(best_task, selected_workers, total_quality, [StrategicArm(mu=worker[f"task_{best_task['id']}_expected_quality"],sigma=worker[f"task_{best_task['id']}_sigma"]).draw() for worker in selected_workers])

    def initialize(self):
        # 空的初始化方法，满足抽象方法的要求
        pass

    def run(self):
        # 运行方法，包括探索阶段和利用阶段
        self.exploration_phase()
        self.exploitation_phase()
        return self.R, self.t
    
    def inspect(self):
        """ for plotting individual rationality """
        self.exploration_phase(True)
        self.exploitation_phase(True)
        return self.R, self.t, self.cp1, self.cp2



if __name__ == "__main__":
    # 主要程序
    num_tasks = 4
    num_workers = 8
    budget = 40
    c_max = 1
    alpha = 1 / 8
    beta = 1 / 8
    tasks = generate_tasks(num_tasks)
    workers = generate_workers(num_workers, tasks)

    # 确保每个任务有一个工人列表
    for task in tasks:
        task['workers'] = workers

    algo = ExplorationAndExploitation(tasks, workers, num_tasks, num_workers, budget, c_max, alpha, beta)
    algo.run()
