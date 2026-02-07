import time
import torch
import jax
import pandas as pd
import matplotlib.pyplot as plt
import os

class BenchmarkTracker:
    def __init__(self):
        self.results = []

    def record(self, framework, mode, batch_size, step_time_ms, peak_mem_mb):
        self.results.append({
            "framework": framework,
            "mode": mode,
            "batch_size": batch_size,
            "latency_ms": step_time_ms,
            "throughput_req_sec": (1000 / step_time_ms) * batch_size,
            "peak_memory_mb": peak_mem_mb
        })

    def save_csv(self, filename="results/benchmark_data.csv"):
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def plot_throughput(self, filename="results/throughput.png"):
        df = pd.DataFrame(self.results)
        plt.figure(figsize=(10, 6))
        for (fw, mode), group in df.groupby(['framework', 'mode']):
            plt.plot(group['batch_size'], group['throughput_req_sec'], marker='o', label=f"{fw} - {mode}")
        
        plt.title("Throughput vs Batch Size (A100)")
        plt.xlabel("Batch Size")
        plt.ylabel("Samples / Sec")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        print(f"Plot saved to {filename}")