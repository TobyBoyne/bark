"""Logging helpers for analysing bottlenecks - not a part of public API."""
from time import perf_counter
from collections import defaultdict

class Timer:
    """Context manager for timing function calls."""
    def __init__(self):
        self.current_key = None
        self.current_start = None
        self.durations = {}

    def __call__(self, key):
        self.current_key = key
        self.current_start = perf_counter()
        return self
    
    def __enter__(self):
        pass

    def __exit__(self, *args):
        num_calls, duration = self.durations.get(self.current_key, (0, 0.0))
        self.durations[self.current_key] = (num_calls + 1, perf_counter() - self.current_start + duration)

    def __getitem__(self, key):
        return self.durations[key]
    
    def __repr__(self):
        return str(self.durations)


class Logger:
    """Logger for recording loss functions over time."""
    def __init__(self):
        self.logs: dict[str, list] = defaultdict(list)

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.logs[key].append(value)

    def __getitem__(self, key):
        return self.logs[key]