class Config:
    num_tasks = 10
    num_workers = 50
    B = 12000
    alpha = 1/10
    beta = 1/8
    c_max = 1
    task_range = [5, 10, 15, 20, 25, 30, 35, 40]  # 任务数范围
    worker_range = [10, 20, 30, 40, 50, 60, 70, 80]  # 工人数范围
    budget_range = [4000,  6000,  8000,  10000, 12000, 14000, 16000, 18000]  # 预算范围

    line_styles = {
        'ExplorationPriorToExploitation': {'color': '#060506', 'marker': 's', 'label': 'ATS-MAB'},
        'ExplorationAndExploitation': {'color': '#ed1e25', 'marker': 'o', 'label': 'AATS-MAB'},
        'Opt': {'color': '#3753a4', 'marker': '^', 'label': 'Opt'},
        'RandomAlgorithm': {'color': '#ffa500', 'marker': 'x', 'label': 'Random'},
    }

    bar_width = 0.15
    bar_styles = {
        'ExplorationPriorToExploitation': {'color': '#060506', 'label': 'ATS-MAB', 'hatch': ''},
        'ExplorationAndExploitation': {'color': '#ed1e25', 'label': 'AATS-MAB', 'hatch': '||'},
        'Opt': {'color': '#3753a4', 'label': 'Opt', 'hatch': '//'},
        'RandomAlgorithm': {'color': '#ffa500', 'label': 'Random', 'hatch': '\\'},
    }
