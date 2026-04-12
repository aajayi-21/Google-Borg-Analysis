import numpy as np


def dimension_weight(tasks_df):
    mean_cpu = tasks_df["cpu"].mean()
    mean_memory = tasks_df["memory"].mean()
    
    cpu_weight = np.exp(0.01 * mean_cpu)
    memory_weight = np.exp(0.01 * mean_memory)

    return cpu_weight, memory_weight



def score_dot_product(task_vector, machine_vector):
    cpu_weight, memory_weight = dimension_weight(task_vector)
    return cpu_weight * task_vector[0] * machine_vector[0] + memory_weight * task_vector[1] * machine_vector[1]


def score_l2norm(task_vector, machine_vector):
    cpu_weight, memory_weight = dimension_weight(task_vector)
    return (cpu_weight * (task_vector[0] - machine_vector[0])**2 
            + memory_weight * (task_vector[1] - machine_vector[1])**2)


def assign_ffd(jobs, machines, fitness_fn):


    

    assignment = {}




    for job in jobs:
        best_machine = None
        best_score = float('inf')
        for machine in machines:
            if machine.fits(job):
                score = fitness_fn(job, machine)
                if score < best_score:
                    best_score = score
                    best_machine = machine
        if best_machine is not None:
            best_machine.assign_task(job)