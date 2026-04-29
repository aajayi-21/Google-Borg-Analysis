import numpy as np

# def dimension_weight(tasks):
#     mean_cpu = tasks["requested_cpus"].mean()
#     mean_memory = tasks["requested_memory"].mean()
    
#     cpu_weight = np.exp(0.01 * mean_cpu)
#     memory_weight = np.exp(0.01 * mean_memory)

#     return cpu_weight, memory_weight

def score_dot_product(task_vector, machine_vector, cpu_weight, memory_weight):
    return cpu_weight * task_vector[0] * machine_vector[0] + memory_weight * task_vector[1] * machine_vector[1]


def score_l2norm(task_vector, machine_vector, cpu_weight, memory_weight):
    return -(cpu_weight * (task_vector[0] - machine_vector[0])**2 
            + memory_weight * (task_vector[1] - machine_vector[1])**2)

def assign_ffd(tasks, machines, fitness_fn, cpu_weight, memory_weight):
    #Note - sequential bin-centric FFD. Explore simultanous assignmnet.
    remaining = list(tasks)   # FFD ordering
    assigned  = set()
    changed   = True

    while changed and remaining:
        changed = False
        for machine in machines:
            if not remaining:
                break
            best_task  = None
            best_score = -np.inf

            for task in remaining:
                if machine.fits(task):
                    score = fitness_fn(task.size, machine.residual_vector(),
                                       cpu_weight, memory_weight)
                    if score > best_score:
                        best_score = score
                        best_task  = task

            if best_task is not None:
                machine.assign_task(best_task)
                assigned.add(best_task)
                remaining.remove(best_task)
                changed = True

    return assigned, set(remaining)


   