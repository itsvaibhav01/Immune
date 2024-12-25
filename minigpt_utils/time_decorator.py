import time
import json
import psutil 
from functools import wraps

execution_stats = {}

def timeit(method):
    @wraps(method)
    def timed(*args, **kwargs):
        start_time = time.time()

        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 ** 2) 

        result = method(*args, **kwargs)

        mem_after = process.memory_info().rss / (1024 ** 2)  
        peak_memory = process.memory_info().vms / (1024 ** 2)  

        end_time = time.time()

        class_name = args[0].__class__.__name__
        method_name = method.__name__
        elapsed_time = end_time - start_time

        if class_name not in execution_stats:
            execution_stats[class_name] = {}
        if method_name not in execution_stats[class_name]:
            execution_stats[class_name][method_name] = {'count': 0, 'times': [], 'memory': []}

        execution_stats[class_name][method_name]['count'] += 1
        execution_stats[class_name][method_name]['times'].append(elapsed_time)
        execution_stats[class_name][method_name]['memory'].append({
            'current': mem_after - mem_before,  # Memory used by the function
            'peak': peak_memory  # Peak memory usage
        })

        return result

    return timed


def save_execution_stats(filename='execution_stats.jsonl'):
    global execution_stats
    with open(filename, 'a') as json_file:
        json_file.write(json.dumps(execution_stats) + '\n')
    
    execution_stats = {}
