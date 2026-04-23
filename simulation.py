import simpy as sp
from task import Task
from machine import Machine
import pandas as pd
import numpy as np
from scoring import assign_ffd, dimension_weight, score_dot_product, score_l2norm
from metrics import compute_metrics
from typing import List, Callable
from tqdm import tqdm

WINDOW = 300_000_000.# 5 minutes in microseconds


def scheduler_process(env:         sp.Environment,
                      task_df:      pd.DataFrame,
                      machines:    List[Machine],
                      fitness_fn:  Callable,
                      metrics_log: list,
                      ):
    """
    Core scheduling loop — one iteration per 5-minute epoch.

    Each epoch
    ----------
    1. Collect all jobs with submit_time ∈ [t, t + WINDOW_SECS).
    2. Build Job objects and sample processing durations.
    3. Run FFD assignment (dot-product or norm-based).
    4. Spawn a concurrent machine_process per assigned job
       (all jobs on a machine execute simultaneously).
    5. Refresh remaining_time on every active job across all machines.
    6. Record window metrics.
    7. Advance the simulation clock by WINDOW_SECS.
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
        window_df = task_df.loc[mask] #New tasks

        assigned   = set()
        remaining = set()

        if not window_df.empty:
            # ── Step 1-2: build Job objects and sample durations ───────────
            #new_tasks = list(remaining) # Remaining tasks from previous window that weren't assigned TODO: Queue is only reduced when new jobs come
            for _, row in window_df.iterrows():
                for _ in range(copies):
                    t = Task(
                        cluster = row["cluster"],
                        collection_id  = row["collection_id"],
                        instance_index = row["instance_index"],
                        cpu            = row["requested_cpus"] * 6,
                        memory         = row["requested_memory"] * 6,
                        submit_time    = env.now,
                        status         = 'SUBMIT'
                    )
                    queue.append(t)
            #cpu_weight, mem_weight = dimension_weight(window_df)
            # ── Step 3: FFD assignment ─────────────────────────────────────
        if queue:
            assigned, remaining = assign_ffd(queue, machines, fitness_fn, 0.5, 0.5)
            queue = list(remaining) # Update queue with remaining tasks that weren't assigned
        #assigned, remaining = assign_ffd(new_tasks, machines, fitness_fn, cpu_weight, mem_weight)
            # ── Step 4: spawn concurrent machine processes ─────────────────
            # Each job gets its own SimPy timeout; jobs on the same machine
            # all run in parallel — there is no internal machine queue.

        for m in machines:  
            if m.tasks: # Only spawn a process if there are tasks assigned to the machine
                env.process(m.process_jobs(env, WINDOW))

        # ── Step 6: record metrics ─────────────────────────────────────────
        rec = compute_metrics(machines, window_id, env.now, assigned, remaining, queue)
        metrics_log.append(rec)

        window_id += 1
        yield env.timeout(WINDOW)
        iteration_count += 1
        progress_bar.update(1)
    progress_bar.close()

#Test commit