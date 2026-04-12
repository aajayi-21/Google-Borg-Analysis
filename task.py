import numpy as np



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
        self.processing_time = self.sample_processing_time(beta0, beta1, beta2,
                                                        stddev, lam)
        self.remining_time = self.processing_time
        self.time_completed = 0


    def sample_processing_time(self, beta0, beta1, beta2, stddev, lam):
        # Sample processing time from a log-normal distribution
        y  = beta0 + beta1 * np.log(self.cpu) + beta2 * np.log(self.memory)
        u = np.random.uniform(0, 1)
        if u <= 0.85:
            err= np.random.normal(0, stddev)
        else:
            err = np.random.exponential(1/lam)
        return np.exp(y + err)


