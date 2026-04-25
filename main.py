from machine import Machine
from task import Task
from metrics import compute_metrics, task_result_df
from scoring import assign_ffd, score_dot_product, score_l2norm
import simpy
import pandas as pd
import numpy as np
from simulation import scheduler_process


def main(task_df, heuristic="dot_product", seed = 42, cpu_weight=0.5, memory_weight=0.5):

    M = 2
    N = len(task_df)
    machines = []
    metrics_log = []
    tasks_log = []

    for machine_id in range(M):
        machines.append(Machine(machine_id, cpu_capacity=1, memory_capacity=1))

    env = simpy.Environment() 
    env.process(scheduler_process(env, task_df, machines, 
                                  fitness_fn=score_dot_product if heuristic == "dot_product" else score_l2norm,
                                  metrics_log=metrics_log, tasks_log=tasks_log, cpu_weight=cpu_weight, 
                                  memory_weight=memory_weight))
    env.run()
    metrics_log_df = pd.DataFrame(metrics_log)
    tasks_log = task_result_df(tasks_log)
    return metrics_log_df, tasks_log




if __name__ == "__main__":
    task_df = pd.read_csv("cleaned_data.csv")
    task_df = task_df[task_df['instance_events_type'] == 'SUBMIT']
    task_df = task_df[(task_df['time'] > 0) & (task_df['time'] < 9223372036854775807)]
    dot_product_metrics_df, dot_prodct_task_df = main(task_df, heuristic="dot_product", seed=42)
    l2norm_metrics_df, l2norm_task_df = main(task_df, heuristic="l2norm", seed=42)
    dot_product_metrics_df.to_csv("dot_product_metrics_log.csv", index=False)
    l2norm_metrics_df.to_csv("l2norm_metrics_log.csv", index=False)
    dot_prodct_task_df.to_csv("dot_product_tasks_log.csv", index=False)
    l2norm_task_df.to_csv("l2norm_tasks_log.csv", index=False)
