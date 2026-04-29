import numpy as np

###CONSTANTS - test
BETA0  = 5.19
BETA1  = 0.30
BETA2  = 0.40
STDDEV = 1.75
LAM    = 1.15



class Task:

    def __init__(self, cluster, collection_id, instance_index, cpu, memory, submit_time, status, rng):
        self.rng = rng
        self.cluster = cluster
        self.collection_id = collection_id
        self.instance_index = instance_index
        self.cpu = cpu
        self.memory = memory
        self.size = np.array([cpu, memory])
        self.submit_time = submit_time
        self.status = status
        self.processing_time = self.sample_processing_time(BETA0, BETA1, BETA2,
                                                        STDDEV, LAM)
        self.remaining_time = self.processing_time
        self.time_completed = np.inf


    def sample_processing_time(self, beta0, beta1, beta2, stddev, lam):
        # Sample processing time from a log-normal distribution
        y  = beta0 + beta1 * np.log(max(self.cpu, 1e-10)) + beta2 * np.log(max(self.memory, 1e-10))
        u = self.rng.uniform(0, 1)
        if u <= 0.85:
            err= self.rng.normal(0, stddev)
        else:
            err = self.rng.exponential(1/lam)
        return np.exp(y + err)


"""Possible Addition
from scipy.stats import norm as sp_norm

def sample_processing_time(self, beta0, beta1, beta2, stddev, alpha):
    
    Sample job processing time in microseconds using a two-component mixture:

      Body (85%) : T | x ~ Lognormal(mu, stddev^2)
                   mu = beta0 + beta1*log(cpu) + beta2*log(memory)

      Tail (15%) : T | x ~ Pareto(x_m=t_85, alpha)  anchored at the 85th
                   percentile of the body so the two components connect
                   exactly with no gap or overlap.

    Pareto inverse CDF:  t = t_min * (1 - u)^(-1/alpha),  u ~ Uniform(0,1)

    Output is in microseconds to match the simulation clock.

    Parameters
    ----------
    alpha : Pareto shape parameter (literature values for Borg: 1.05–1.30)
            Lower alpha = heavier tail.  NOT an exponential rate.
    
    mu = (beta0
          + beta1 * np.log(max(float(self.cpu),    1e-10))
          + beta2 * np.log(max(float(self.memory), 1e-10)))

    u = rng.uniform(0.0, 1.0)

    if u <= 0.85:
        # ── Lognormal body ─────────────────────────────────────────────────
        t_seconds = np.exp(mu + rng.normal(0.0, stddev))

    else:
        # ── Pareto tail ────────────────────────────────────────────────────
        # Anchor point: 85th percentile of the body Lognormal.
        # sp_norm.ppf(0.85) ≈ 1.036
        t_min = np.exp(mu + stddev * sp_norm.ppf(0.85))

        # Inverse CDF of Pareto:  t = t_min * (1 - v)^(-1/alpha)
        v = rng.uniform(0.0, 1.0)
        t_seconds = t_min * (1.0 - v) ** (-1.0 / alpha)

    # ── Convert seconds → microseconds (simulation clock unit) ────────────
    return max(t_seconds * 1_000_000, 1.0)   # floor at 1 µs
"""