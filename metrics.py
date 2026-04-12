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


def _state_diversity(machines: List[Machine], n_bins: int = 10) -> float:
    """
    Diversity of residual capacity states across machines.

    Procedure
    ---------
    1. Discretize each machine's CPU and memory *utilization* into one of
       n_bins buckets:  bin_index = floor(utilization * n_bins),
       clamped to [0, n_bins - 1] to handle utilization == 1.0 exactly.
    2. Count A(i, j) = number of machines whose state falls in bucket (i, j).
    3. Return  D = sum_{i,j} A(i,j)^2   (lower = more diverse).

    Bounds
    ------
      Minimum : M           (every machine in a distinct bucket, A(i,j) ≤ 1)
      Maximum : M^2         (every machine in the same bucket)
      Uniform : M^2 / B     (M machines spread evenly across B occupied buckets)

    Parameters
    ----------
    n_bins : number of discretization steps per dimension.
             n_bins=10 → 10 % increments, 100 possible buckets.
             n_bins=20 → 5  % increments, 400 possible buckets.
    """
    counts: Dict[tuple, int] = {}

    for m in machines:
        r = m.residual_vector()                        # [cpu_residual, mem_residual]
        cpu_util = 1.0 - r[0] / m.cpu_capacity        # fraction of CPU used
        mem_util = 1.0 - r[1] / m.memory_capacity     # fraction of memory used

        i = int(min(cpu_util * n_bins, n_bins - 1))   # clamp so 100% → last bin
        j = int(min(mem_util * n_bins, n_bins - 1))

        counts[(i, j)] = counts.get((i, j), 0) + 1

    return float(sum(v ** 2 for v in counts.values()))


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
        "state_diversity":      _state_diversity(machines, n_bins=10),
        "state_diversity_norm": _state_diversity(machines, n_bins=10) / (len(machines) ** 2),
    } 