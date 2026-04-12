from machine import Machine
from task import Task
from metrics import compute_metrics
from scoring import assign_ffd, score_dot_product, score_l2norm
import simpy
import pandas as pd
import numpy as np
from simulation import scheduler_process


def main(task_df, heuristic="dot_product", seed = 42):

    M = 2
    N = len(task_df)
    machines = []
    metrics_log = []

    for machine_id in range(M):
        machines.append(Machine(machine_id, cpu_capacity=1, memory_capacity=1))

    env = simpy.Environment() 
    env.process(scheduler_process(env, task_df, machines, 
                                  fitness_fn=score_dot_product if heuristic == "dot_product" else score_l2norm,
                                  metrics_log=metrics_log))
    env.run()
    return pd.DataFrame(metrics_log)


if __name__ == "__main__":
    task_df = pd.read_csv("cleaned_data.csv")
    task_df = task_df[task_df['instance_events_type'] == 'SUBMIT']
    task_df = task_df[(task_df['time'] > 0) & (task_df['time'] < 9223372036854775807)]
    metrics_df = main(task_df, heuristic="dot_product", seed=42)
    metrics_df.to_csv("metrics_log.csv", index=False)
