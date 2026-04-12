import pandas as pd 
import numpy as np
from typing import List, Set, Dict
from machine import Machine

def _gini(values: np.ndarray) -> float:
    """Gini coefficient — measures load (im)balance across machines."""
    a = np.sort(np.abs(values))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum() - (n + 1) * a.sum()) / (n * a.sum()))


def compute_metrics(machines:    List[Machine],
                    window_id:   int,
                    sim_time:    float,
                    assigned: Set,
                    remaining:  Set) -> Dict:
    """
    Compute per-window system performance metrics.

    remaining_time on each task has been refreshed by scheduler_process before
    this function is called, so all metrics reflect the live in-progress state.

    Utilization
      avg_load         — mean (cpu+mem)/2 utilization across all machines
      max_load         — load on the most-loaded machine
      cpu_utilization  — mean CPU utilization
      mem_utilization  — mean memory utilization

    Fragmentation / slack
      slack_variance   — variance of per-machine L2 slack norms
      slack_gini       — Gini coefficient of slack  (balance metric)
      avg_slack_l2     — mean residual L2 norm across machines

    Schedulability
      admission_rate   — fraction of submitted tasks successfully assigned
      rejection_prob   — 1 − admission_rate

    In-progress workload
      active_tasks      — total tasks still running across all machines
      avg_remaining    — mean remaining_time (s) of all active tasks
      max_remaining    — maximum remaining_time (s) across active tasks
    """
    cpu_loads = np.array([1.0 - m.residual_vector()[0] for m in machines])
    mem_loads = np.array([1.0 - m.residual_vector()[1] for m in machines])
    avg_cpu_load = np.mean(cpu_loads)
    avg_mem_load = np.mean(mem_loads)

    admit_rate = (len(assigned) / (len(assigned) + len(remaining))) if (len(assigned) + len(remaining)) > 0 else 0.0 

    slack_l2 = np.array([np.linalg.norm(m.residual_vector()) for m in machines])


    loads_cpu = np.array([1.0 - m.residual_vector()[0] for m in machines])
    loads_mem = np.array([1.0 - m.residual_vector()[1] for m in machines])
   
    slack_l2  = np.array([np.linalg.norm(m.residual_vector()) for m in machines])

    all_active_tasks = [task for m in machines for task in m.tasks]
    remaining_times = np.array([task.remaining_time for task in all_active_tasks])


    return {
        "window_id":       window_id,
        "sim_time":        sim_time,
        # Utilization
        "avg_cpu_load":    float(avg_cpu_load),
        "avg_mem_load":    float(avg_mem_load),
        "max_cpu_load":    float(np.max(cpu_loads)),
        "max_mem_load":    float(np.max(mem_loads)),
        "var_cpu_load":    float(np.var(cpu_loads)),
        "var_mem_load":    float(np.var(mem_loads)),
        # Fragmentation / slack
        "slack_variance":  float(np.var(slack_l2)),
        "slack_gini":      _gini(slack_l2),
        "avg_slack_l2":    float(np.mean(slack_l2)),
        # Schedulability
        "admission_rate":  admit_rate,
        "rejection_prob":  1.0 - admit_rate,
        # In-progress workload
        "num_bins_used":   int(sum(1 for m in machines if len(m.tasks) > 0)),
        #"active_tasks":     len(all_active_tasks),
        #"avg_remaining":   float(np.mean(remaining_times)),
        "n_submitted":     len(assigned) + len(remaining),
        "n_assigned":      len(assigned),
    } 