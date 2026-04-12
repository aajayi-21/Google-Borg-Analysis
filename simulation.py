import simpy as sp
from task import Task
from machine import Machine
import pandas as pd
import numpy as np


def scheduler_process(env:         sp.Environment,
                      task_df:      pd.DataFrame,
                      machines:    List[Machine],
                      fitness_fn:  Callable,
                      metrics_log: list,
                      rng:         np.random.Generator,
                      heuristic:   str):
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
    trace_end = task_df["time"].max()

    while env.now <= trace_end + WINDOW:
        t_start = env.now
        t_end   = env.now + WINDOW

        mask      = ((task_df["time"] >= t_start) &
                    (task_df["time"] <  t_end))
        window_df = task_df.loc[mask] #New tasks

        n_submitted = len(window_df)
        n_assigned  = 0

        if not window_df.empty:
            # ── Step 1-2: build Job objects and sample durations ───────────
            new_tasks = []
            for _, row in window_df.iterrows():
                t = Task(
                    cluster = row["cluster"],
                    collection_id  = row["collection_id"],
                    instance_index = row["instance_index"],
                    cpu            = row["cpu"],
                    memory         = row["memory"],
                    submit_time    = env.now,
                    status         = 'SUBMIT'
                )
                new_tasks.append(t)

            # ── Step 3: FFD assignment ─────────────────────────────────────
            assignment = assign_ffd(new_tasks, machines, fitness_fn)
            n_assigned = len(assignment)

            # ── Step 4: spawn concurrent machine processes ─────────────────
            # Each job gets its own SimPy timeout; jobs on the same machine
            # all run in parallel — there is no internal machine queue.

            for m in machines:  
                m.process_jobs(env, WINDOW)

        # ── Step 6: record metrics ─────────────────────────────────────────
        rec = compute_metrics(machines, window_id, env.now,
                              heuristic, n_submitted, n_assigned)
        metrics_log.append(rec)

        window_id += 1
        yield env.timeout(WINDOW_SECS)