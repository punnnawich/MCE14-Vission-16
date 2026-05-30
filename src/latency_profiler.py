import time
from collections import deque
import numpy as np

class LatencyProfiler:
    def __init__(self, window_size=30):
        """
        Initialize profiler to track stage latencies with a sliding window.
        """
        self.window_size = window_size
        self.stages = {}
        self.history = {}  # stage_name -> deque of latencies (ms)
        self.frame_start = None

    def start_frame(self):
        """
        Mark the start of a pipeline iteration.
        """
        self.frame_start = time.perf_counter()
        self.stages.clear()

    def start_stage(self, stage_name):
        """
        Start timing a stage.
        """
        self.stages[stage_name] = time.perf_counter()

    def end_stage(self, stage_name):
        """
        End timing a stage and record its duration.
        """
        if stage_name in self.stages:
            duration_ms = (time.perf_counter() - self.stages[stage_name]) * 1000.0
            if stage_name not in self.history:
                self.history[stage_name] = deque(maxlen=self.window_size)
            self.history[stage_name].append(duration_ms)

    def end_frame(self):
        """
        Mark the end of the pipeline iteration and record total pipeline duration.
        """
        if self.frame_start is not None:
            duration_ms = (time.perf_counter() - self.frame_start) * 1000.0
            stage_name = "Total Pipeline"
            if stage_name not in self.history:
                self.history[stage_name] = deque(maxlen=self.window_size)
            self.history[stage_name].append(duration_ms)

    def get_averages(self):
        """
        Returns average duration (ms) for all timed stages.
        """
        averages = {}
        for name, deq in self.history.items():
            if len(deq) > 0:
                averages[name] = np.mean(deq)
        return averages

    def get_latest(self):
        """
        Returns the latest duration (ms) for all timed stages.
        """
        latest = {}
        for name, deq in self.history.items():
            if len(deq) > 0:
                latest[name] = deq[-1]
        return latest

    def print_summary(self):
        """
        Prints a formatted timing summary table to the console.
        """
        averages = self.get_averages()
        print("\n--- Latency Profiler Summary (ms) ---")
        print(f"{'Stage Name':<35} | {'Average (ms)':<12} | {'Latest (ms)':<12}")
        print("-" * 65)
        for name in sorted(averages.keys()):
            if name == "Total Pipeline":
                continue
            avg_val = averages[name]
            lat_val = self.history[name][-1]
            print(f"{name:<35} | {avg_val:<12.2f} | {lat_val:<12.2f}")
        print("-" * 65)
        if "Total Pipeline" in averages:
            print(f"{'Total Pipeline':<35} | {averages['Total Pipeline']:<12.2f} | {self.history['Total Pipeline'][-1]:<12.2f}")
        print("-------------------------------------\n")
