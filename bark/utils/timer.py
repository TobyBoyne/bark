"""Timer used for keeping track of the time spent in fitting and optimizing."""

from time import perf_counter

class Timer(dict):
    def __init__(self):
        self.start_time = 0
        self.current_key = ""

    def __call__(self, key: str):
        self.current_key = key
        return self

    def __enter__(self):
        self.start_time = perf_counter()
        return self
    
    def __exit__(self, *args):
        total_time = self.get(self.current_key, 0)
        self[self.current_key] = total_time + perf_counter() - self.start_time
        self.current_key = ""
