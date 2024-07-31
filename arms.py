import random

class NormalArm:
    def __init__(self, mu: float, sigma: float):
        """ Mean and standard deviation for the normal distribution."""
        self.mu = mu
        self.sigma = sigma

    def draw(self):
        """ Returns the achieved reward of the arm at this round. """
        return random.gauss(self.mu, self.sigma)

class StrategicArm(NormalArm):
    c_min, c_max = 0.1, 1

    def __init__(self, mu: float = None, sigma: float = None):
        if mu is not None and sigma is not None:
            super().__init__(mu, sigma)
        else:
            # in the paper, r is expected reward
            r = random.uniform(0.1, 1)
            # to make that sample value is within 0~1 with 97%
            sigma = random.uniform(0, min(r / 3, (1 - r) / 3))
            super().__init__(r, sigma)

        # c for cost, b for bid, c_i = b_i according to the theorem 2
        self.c = random.uniform(self.c_min, self.c_max)
        self.b = self.c

def generate_tasks(num_tasks, k_min=1, k_max=3):
    tasks = []
    for i in range(num_tasks):
        required_workers = random.randint(k_min, k_max)
        expected_reward = required_workers * random.uniform(0.1, 1)  # 每个工人贡献0到0.1的奖励
        task = {
            "id": i + 1,
            "required_workers": required_workers,
            "expected_reward": expected_reward,
            "workers": []
        }
        tasks.append(task)
    return tasks

def generate_workers(num_workers, tasks):
    workers = []
    for i in range(num_workers):
        worker = {"id": i + 1}
        for task in tasks:
            # 工人的预期质量应该使得所需工人数的工人总质量在任务预期奖励附近波动
            quality_per_worker = task["expected_reward"] / task["required_workers"] * random.uniform(0.8, 1.2)
            sigma_per_worker = random.uniform(0, min(quality_per_worker / 3, (1 - quality_per_worker) / 3))
            arm = StrategicArm(mu=quality_per_worker, sigma=sigma_per_worker)  # 直接设定mu为计算的质量
            worker[f"task_{task['id']}_bid"] = arm.b
            worker[f"task_{task['id']}_expected_quality"] = arm.mu
            worker[f"task_{task['id']}_sigma"] = arm.sigma
        workers.append(worker)
    return workers

if __name__ == "__main__":
    # Example usage
    num_tasks = 2
    num_workers = 3
    tasks = generate_tasks(num_tasks)
    workers = generate_workers(num_workers, tasks)
    print("Tasks:", tasks)
    print("Workers:", workers)
