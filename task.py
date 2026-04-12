import numpy as np

###CONSTANTS - test
BETA0  = 5.19
BETA1  = 0.30
BETA2  = 0.40
STDDEV = 1.75
LAM    = 1.15

rng = np.random.default_rng(42)

class Task:

    def __init__(self, cluster, collection_id, instance_index, cpu, memory, submit_time, status):
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
        self.time_completed = 0


    def sample_processing_time(self, beta0, beta1, beta2, stddev, lam):
        # Sample processing time from a log-normal distribution
        y  = beta0 + beta1 * np.log(max(self.cpu, 1e-10)) + beta2 * np.log(max(self.memory, 1e-10))
        u = rng.uniform(0, 1)
        if u <= 0.85:
            err= rng.normal(0, stddev)
        else:
            err = rng.exponential(1/lam)
        return np.exp(y + err)


