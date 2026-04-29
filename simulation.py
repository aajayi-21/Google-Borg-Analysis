import simpy as sp
from task import Task
from machine import Machine
import pandas as pd
import numpy as np
from scoring import assign_ffd, score_dot_product, score_l2norm
from metrics import compute_metrics
from typing import List, Callable
from tqdm import tqdm

WINDOW = 120_000_000.# 2 minutes in microseconds


def task_process(env, task, machine):
    """
    A process that represents a single task's lifecycle on a machine.
    """
    # Wait for the exact duration (converted to microseconds)
    duration_micros = task.processing_time * 1_000_000
    yield env.timeout(duration_micros)
    
    # Task is finished
    task.time_completed = env.now
    task.remaining_time = 0
    machine.release_task(task)


def scheduler_process(env:         sp.Environment,
                    task_df:      pd.DataFrame,
                    machines:    List[Machine],
                    fitness_fn:  Callable,
                    metrics_log: list,
                    tasks_log: list,
                    cpu_weight: float = 0.5,
                    memory_weight: float = 0.5,
                    rng = None
                    ):
    """
    Core scheduling loop — one iteration per 2-minute epoch.

    Each epoch
    ----------
    1. Collect all jobs with submit_time ∈ [t, t + WINDOW_SECS).
    2. Build Job objects and sample processing durations.
    3. Run assignment (dot-product or norm-based).
    4. Spawn an exact-duration task_process for each assigned job.
    5. Record window metrics.
    6. Advance the simulation clock by WINDOW_SECS.
    """
    window_id = 0
    copies = 1
    trace_end = task_df["time"].max()
    MAX_ITER = int(trace_end / WINDOW + 1) + 1
    iteration_count = 0 
    progress_bar = tqdm(total=MAX_ITER, desc="Simulation Progress", unit="epoch")


    queue = []
    while env.now <= trace_end + WINDOW:
        t_start = env.now
        t_end   = env.now + WINDOW

        mask      = ((task_df["time"] >= t_start) &
                    (task_df["time"] <  t_end))
        window_df = task_df.loc[mask] 

        assigned   = set()
        remaining = set()

        if not window_df.empty:
            # ── Step 1-2: build Job objects and sample durations ───────────
            for _, row in window_df.iterrows():
                for _ in range(copies):
                    t = Task(
                        cluster = row["cluster"],
                        collection_id  = row["collection_id"],
                        instance_index = row["instance_index"],
                        cpu            = row["requested_cpus"] * 10,
                        memory         = row["requested_memory"] * 10,
                        submit_time    = env.now,
                        status         = 'SUBMIT',
                        rng = rng
                    )
                    queue.append(t)
                    tasks_log.append(t)
            #cpu_weight, mem_weight = dimension_weight(window_df)
            # ── Step 3: FFD assignment ─────────────────────────────────────
        if queue:
            assigned, remaining = assign_ffd(queue, machines, fitness_fn, cpu_weight=cpu_weight, 
                                            memory_weight=memory_weight)
            queue = list(remaining) 
            
            # ── Step 4: spawn exact-duration task processes ────────────────
            for task in assigned:
                env.process(task_process(env, task, task.machine))

        # Advance simulation clock by WINDOW
        yield env.timeout(WINDOW)

        # ── Step 5: record metrics AFTER the window has passed ─────────────
        # This captures the state of the system at the end of the interval
        rec = compute_metrics(machines, window_id, env.now, assigned, remaining, queue)
        metrics_log.append(rec)

        window_id += 1
        iteration_count += 1
        progress_bar.update(1)
    progress_bar.close()

#Test commit