# binpack_env.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# You will import your own modules here
# from scheduler import run_one_window   # you need to implement this wrapper
# from metrics import compute_metrics    # already exist in your code


class BinPackWeightEnv(gym.Env):
    """
    Gymnasium environment for learning heuristic + dimension weights.

    At each step (5-minute window), the agent chooses:
      - which heuristic to use (e.g., dot vs L2),
      - CPU and memory weights in [0, 1] (later normalized).

    The environment runs one scheduling window using your SimPy-based simulator,
    then returns aggregated metrics as observation and a scalar reward.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 200,
        # You can pass additional args to configure the simulator:
        sim_config: Dict[str, Any] | None = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.sim_config = sim_config or {}

        # ----- Action space -----
        # action[0]: heuristic choice (0=dot, 1=L2, 2=hybrid, etc.)
        # action[1]: cpu_weight in [0,1]
        # action[2]: mem_weight in [0,1]
        self.n_heuristics = 2  # start with 2, extend as needed
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([float(self.n_heuristics) - 1e-3, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ) #TODO: Mak sure they sum to 1.0

        # ----- Observation space -----
        # Example observation vector (you can extend):
        # [avg_cpu_load, avg_mem_load, slack_gini,
        #  state_diversity_norm, queue_length_norm, admission_rate]
        # All normalized to [0, 1] where possible.
        #The observtion space should be the entire queue and machine states, but we can start with a fixed-size summary vector and iterate.
        self.obs_dim = 6
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Internal simulation state
        self.current_step = 0
        self.sim = None  # placeholder for your SimPy + cluster state

    # ---------- Helpers to hook into your simulator ----------

    def _init_sim(self, seed: int | None = None):
        """
        Create / reset your SimPy environment, machines, and task trace iterator.
        This should set up everything so that one call to _run_one_window()
        advances exactly one WINDOW and returns metrics.
        """
        # You need to implement this according to your project structure.
        # Example:
        # self.sim = make_simpy_env(task_df, machines, WINDOW, seed=seed, **self.sim_config)
        rng = np.random.default_rng(seed)
        self.sim 
        raise NotImplementedError("_init_sim must be implemented to set up your simulator.")

    def _run_one_window(
        self,
        heuristic_idx: int,
        cpu_weight: float,
        mem_weight: float,
    ) -> Dict[str, float]:
        """
        Advance the simulation by exactly one window using the chosen heuristic + weights.

        Returns:
            metrics: dict containing at least
                - avg_cpu_load
                - avg_mem_load
                - slack_gini
                - state_diversity_norm
                - queue_length
                - admission_rate
        """
        # This should:
        #  1. Map heuristic_idx -> fitness_fn (e.g. score_dot_product / score_l2norm / hybrid)
        #  2. Run the assignment for this window with assign_ffd(..., fitness_fn, cpu_weight, mem_weight)
        #  3. Let SimPy advance by WINDOW and update machines/tasks
        #  4. Call compute_metrics(...) for this window and return the dict
        raise NotImplementedError("_run_one_window must be implemented using your scheduler.")

    def _metrics_to_obs(self, metrics: Dict[str, float]) -> np.ndarray:
        """
        Convert raw metrics dict to a normalized observation vector.
        """
        avg_cpu = np.clip(metrics.get("avg_cpu_load", 0.0), 0.0, 1.0)
        avg_mem = np.clip(metrics.get("avg_mem_load", 0.0), 0.0, 1.0)
        slack_gini = np.clip(metrics.get("slack_gini", 0.0), 0.0, 1.0)

        # state_diversity_norm is already in [0,1] in your code
        state_div_norm = np.clip(metrics.get("state_diversity_norm", 0.0), 0.0, 1.0)

        # Queue length normalization: you may want to divide by a fixed max_queue
        max_queue = 1000.0  # choose a reasonable constant for your setting
        queue_len_norm = np.clip(metrics.get("queue_length", 0.0) / max_queue, 0.0, 1.0)

        admit_rate = np.clip(metrics.get("admission_rate", 0.0), 0.0, 1.0)

        obs = np.array(
            [
                avg_cpu,
                avg_mem,
                slack_gini,
                state_div_norm,
                queue_len_norm,
                admit_rate,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self, metrics: Dict[str, float]) -> float:
        """
        Scalar reward from metrics. You can tune this.
        Example: encourage utilization and admission, penalize fragmentation and queueing.
        """
        avg_cpu = metrics.get("avg_cpu_load", 0.0)
        avg_mem = metrics.get("avg_mem_load", 0.0)
        admit_rate = metrics.get("admission_rate", 0.0)
        slack_gini = metrics.get("slack_gini", 0.0)
        state_div_norm = metrics.get("state_diversity_norm", 0.0)
        queue_len = metrics.get("queue_length", 0.0)

        util = 0.5 * (avg_cpu + avg_mem)

        # Normalize queue and diversity using simple scaling factors
        queue_pen = queue_len / 1000.0   # adjust denominator as needed

        reward = (
            2.0 * admit_rate               # strongly reward high admission
            + 1.0 * util                   # reward utilization
            - 0.5 * slack_gini             # penalize imbalance
            - 0.5 * state_div_norm         # penalize fragmented states
            - 0.5 * queue_pen              # penalize long queues
        )
        return float(reward)

    # ---------- Gymnasium API ----------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = 0
        self._init_sim(seed=seed)

        # Optionally run an initial window with default heuristic/weights to generate first obs,
        # or start with a "null" metrics state.
        metrics0 = {
            "avg_cpu_load": 0.0,
            "avg_mem_load": 0.0,
            "slack_gini": 0.0,
            "state_diversity_norm": 0.0,
            "queue_length": 0.0,
            "admission_rate": 1.0,
        }
        obs = self._metrics_to_obs(metrics0)
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        action: np.array of shape (3,)
            [heuristic_idx_float, cpu_weight_raw, mem_weight_raw]
        """
        self.current_step += 1

        # Parse action
        action = np.asarray(action, dtype=np.float32)
        heuristic_idx = int(np.clip(round(action[0]), 0, self.n_heuristics - 1))

        cpu_w_raw = float(np.clip(action[1], 0.0, 1.0))
        mem_w_raw = float(np.clip(action[2], 0.0, 1.0))

        # Normalize weights so they sum to 1 (avoid degenerate all-zero)
        denom = cpu_w_raw + mem_w_raw
        if denom <= 1e-6:
            cpu_weight = 0.5
            mem_weight = 0.5
        else:
            cpu_weight = cpu_w_raw / denom
            mem_weight = mem_w_raw / denom

        # Run one epoch/window in your simulator
        metrics = self._run_one_window(
            heuristic_idx=heuristic_idx,
            cpu_weight=cpu_weight,
            mem_weight=mem_weight,
        )

        obs = self._metrics_to_obs(metrics)
        reward = self._compute_reward(metrics)

        # Termination condition: end of trace or max_steps reached.
        terminated = bool(metrics.get("sim_finished", False))
        truncated = self.current_step >= self.max_steps

        info = {
            "metrics": metrics,
            "heuristic_idx": heuristic_idx,
            "cpu_weight": cpu_weight,
            "mem_weight": mem_weight,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # Simple text rendering of last metrics or step
            print(f"Step {self.current_step}")

    def close(self):
        self.sim = None