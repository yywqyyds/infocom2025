import os
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from arms import generate_tasks, generate_workers
from config import Config
from experiment import Emulator
import pickle
import scienceplots
import matplotlib

matplotlib.rcParams['text.usetex'] = False

# plt.style.use(['science', 'grid'])
config = Config

# data preparation
data = []
# 遍历任务数、工人数和预算范围
# for X in ['task', 'worker', 'budget']:
for X in ['worker']:
    for x in tqdm(eval(f'config.{X}_range'), desc=X):
        # 生成任务和工人

        # 根据不同的X设置模拟器
        if X == 'task':
            tasks = generate_tasks(x)
            workers = generate_workers(config.num_workers, tasks)
            for task in tasks:
                task['workers'] = workers
            name2res = Emulator(tasks, workers, num_tasks=x, num_workers=config.num_workers, budget=config.B,
                                c_max=config.c_max, alpha=config.alpha, beta=config.beta).simulate()
        elif X == 'worker':
            tasks = generate_tasks(config.num_tasks)
            workers = generate_workers(x, tasks)
            for task in tasks:
                task['workers'] = workers
            name2res = Emulator(tasks, workers, num_tasks=config.num_tasks, num_workers=x, budget=config.B,
                                c_max=config.c_max, alpha=config.alpha, beta=config.beta).simulate()
        else:
            tasks = generate_tasks(config.num_tasks)
            workers = generate_workers(config.num_workers, tasks)
            for task in tasks:
                task['workers'] = workers
            name2res = Emulator(tasks, workers, num_tasks=config.num_tasks, num_workers=config.num_workers,
                                budget=x, c_max=config.c_max, alpha=config.alpha, beta=config.beta).simulate()

        for key in name2res.keys():
            data.append([X, x, key, name2res[key][0], name2res[key][1]])

df = pd.DataFrame(np.array(data), columns=['X', 'Val', 'Algorithm', 'Reward', 'Round'])

with open('./runs.pkl', 'wb') as fout:
    pickle.dump(df, fout)

# 结果可视化
def plot_results(df):
    print("Starting plot_results...")
    print(df.head())
    df['Val'] = df['Val'].astype(float)
    df['Reward'] = df['Reward'].astype(float)
    df['Round'] = df['Round'].astype(float)

    # Custom formatter to display y-axis in terms of 10^4
    from matplotlib.ticker import FuncFormatter

    def y_fmt(x, pos):
        return '{:.1f}'.format(x / 1e4)
    def y_fmt1(x, pos):
        return '{:.1f}'.format(x / 1e4)

    # 折线图
    fig, ax = plt.subplots(1, 1, figsize=(12, 6.5))
    # plt.legend(fontsize="20")
    plt.rcParams['xtick.labelsize'] = 48
    plt.rcParams['ytick.labelsize'] = 48
    for algo in Emulator.algorithms:
        data = df[(df.X == 'budget') & (df.Algorithm == algo)]
        if not data.empty:
            ax.plot(data['Val'], data['Reward'], **config.line_styles[algo])
            ax.set_xlabel('Budget', fontsize=28)
            ax.set_ylabel('Total rewards ($\\times 10^4$)', fontsize=28)
            ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
            ax.legend(fontsize=28)
            plt.savefig(f'B-R-W{config.num_workers}-T{config.num_workers}.pdf', bbox_inches='tight')

            # ax = axes[0, 1]
            # ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
            # ax.set_xlabel('Budget')
            # ax.set_ylabel('Total rounds')

        # data = df[(df.X == 'task') & (df.Algorithm == algo)]
        # if not data.empty:
        #     ax = axes[1, 1]
        #     ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
        #     ax.set_xlabel('Number of tasks')
        #     ax.set_ylabel('Total rounds')
        #
        # data = df[(df.X == 'worker') & (df.Algorithm == algo)]
        # if not data.empty:
        #     ax = axes[2, 1]
        #     ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
        #     ax.set_xlabel('Number of workers')
        #     ax.set_ylabel('Total rounds')

    # 柱状图
    n_algos = len(Emulator.algorithms)

    X = 'task'

    data = df[df.X == X].pivot(index='Val', columns='Algorithm', values='Reward')
    if not data.empty:
        for i, algo in enumerate(Emulator.algorithms):
            xpos = np.arange(len(data.index)) + (i - n_algos / 2) * config.bar_width
            ax.bar(xpos, data[algo], width=config.bar_width, **config.bar_styles[algo])

        ax.set_ylabel('Total rewards ($\\times 10^3$)')
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index)
        ax.yaxis.set_major_formatter(FuncFormatter(y_fmt1))

        ax.set_xlabel('Number of tasks')
        ax.legend()
        plt.savefig(f'T-R-W{config.num_workers}-B{config.B}.pdf', bbox_inches='tight')

    X = 'worker'
    data = df[df.X == X].pivot(index='Val', columns='Algorithm', values='Reward')
    if not data.empty:
        for i, algo in enumerate(Emulator.algorithms):
            xpos = np.arange(len(data.index)) + (i - n_algos / 2) * config.bar_width
            ax.bar(xpos, data[algo], width=config.bar_width, **config.bar_styles[algo])

        ax.set_ylabel('Total rewards ($\\times 10^3$)', fontsize=28)
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index)
        ax.yaxis.set_major_formatter(FuncFormatter(y_fmt1))
        ax.set_xlabel('Number of workers', fontsize=28)
        ax.legend(fontsize=28)
        plt.savefig(f'W-R-T{config.num_tasks}-B{config.B}.pdf', bbox_inches='tight')
    plt.close()
    #
    # plt.savefig('fig.pdf', dpi=800)

# 运行模拟
if __name__ == "__main__":

    df = pd.read_pickle('./runs.pkl')
    print(df)
    plot_results(df)
