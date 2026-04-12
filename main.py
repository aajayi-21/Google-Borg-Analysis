from machine import Machine
from task import Task
from metrics import compute_metrics
from scoring import assign_ffd, score_dot_product, score_norm
import simpy
import pandas as pd
import numpy as np
from simulation import scheduler_process

###CONSTANTS
WINDOW = 300000000 # 5 minutes in microseconds
BETA0 = 0
BETA1 = 0.5
BETA2 = 0.5
STDDEV = 0.1
LAM = 1



def main():
    task_df = pd.read_csv("task_df.csv")
    M = task_df["machine_id"].nunique()
    machines = []
    for machine_id in range(M):
        machines.append(Machine(machine_id, cpu_capacity=1, memory_capacity=1))

    env = simpy.Environment() 
