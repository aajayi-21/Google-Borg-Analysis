from task import Task
import numpy as np

class Machine:
    def __init__(self, machine_id, cpu_capacity, memory_capacity):
        self.machine_id = machine_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.cpu_used = 0
        self.memory_used = 0
        self.available_cpu = cpu_capacity
        self.available_memory = memory_capacity
        self.tasks = []

    def cpu_residual(self):
        return self.cpu_capacity - self.cpu_used
    
    def memory_residual(self):
        return self.memory_capacity - self.memory_used
    
    def fits(self, task):
        return (self.available_cpu >= task.cpu 
                and self.available_memory >= task.memory)
    
    def residual_vector(self):
        return np.array([self.cpu_residual(), self.memory_residual()])
    
    def assign_task(self, task):
        if self.fits(task):
            task.status = "SCHEDULE"
            task.machine = self
            self.tasks.append(task)
            self.cpu_used += task.cpu
            self.memory_used += task.memory
            self.available_cpu -= task.cpu
            self.available_memory -= task.memory
            return True
        else:
            return False
        
    def release_task(self, task):
        if task in self.tasks:
            task.status='FINISH'
            self.tasks.remove(task)
            self.cpu_used -= task.cpu
            self.memory_used -= task.memory
            self.available_cpu += task.cpu
            self.available_memory += task.memory
            return True
        else:
            return False
        
    def process_jobs(self, env, window):
        num_completed = 0
        yield env.timeout(window)
        window_sec = window / 1000000
        for task in list(self.tasks):
            task.remaining_time -= window_sec
            if task.remaining_time <= 0:
                task.remaining_time = 0
                task.time_completed = env.now
                self.release_task(task)


