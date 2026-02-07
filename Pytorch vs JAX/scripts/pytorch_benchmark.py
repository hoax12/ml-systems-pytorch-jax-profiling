import torch
from transformers import DistilBertForSequenceClassification
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.baseline import BATCH_SIZES, NUM_STEPS, WARMUP_STEPS, get_dummy_data_torch, MODEL_ID
from profiling.profiler_utils import BenchmarkTracker

def train_loop(model, batch_size, optimizer, device, enable_compile=False, capture_trace=False):
    model.train()
    
    # 1. Setup Data
    input_ids, attn_mask, labels = get_dummy_data_torch(batch_size)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    labels = labels.to(device)

    # 2. Compile FIRST (so we can profile the compiled model too)
    if enable_compile:
        print(f"Compiling model (Batch {batch_size})...")
        model = torch.compile(model, mode="reduce-overhead")

    # Define the training step function to avoid code duplication
    def run_step():
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()

    # 3. Profiling Mode (Trace Capture)
    if capture_trace:
        print("Capturing TensorBoard trace...")
        # Create results folder if missing
        os.makedirs('./results/traces', exist_ok=True)
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./results/traces'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            # Run enough steps to satisfy the schedule (1 wait + 1 warmup + 3 active = 5 steps minimum)
            for i in range(10): 
                run_step()  # <--- CRITICAL: Actual training logic here
                prof.step()
        
        # Return dummy values since we are only capturing traces
        return 0.0, 0.0

    # 4. Benchmarking Mode (Latency/Memory)
    else:
        # Warmup
        for _ in range(WARMUP_STEPS):
            run_step()
        
        # Reset Memory Stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Start Timer
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(NUM_STEPS):
            run_step()
        end_event.record()
        
        torch.cuda.synchronize()
        
        total_time = start_event.elapsed_time(end_event) # ms
        avg_step_time = total_time / NUM_STEPS
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return avg_step_time, peak_mem

def run_benchmarks():
    tracker = BenchmarkTracker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Eager Mode
    print("--- Running PyTorch Eager ---")
    for bs in BATCH_SIZES:
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        t, mem = train_loop(model, bs, opt, device, enable_compile=False)
        tracker.record("PyTorch", "Eager", bs, t, mem)
        print(f"Batch {bs}: {t:.2f}ms/step, {mem:.0f}MB")

    # Compile Mode
    print("\n--- Running PyTorch Compile ---")
    for bs in BATCH_SIZES:
        # Re-init model to clear cache
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        t, mem = train_loop(model, bs, opt, device, enable_compile=True)
        tracker.record("PyTorch", "Compile", bs, t, mem)
        print(f"Batch {bs}: {t:.2f}ms/step, {mem:.0f}MB")

    tracker.save_csv()
    tracker.plot_throughput()

if __name__ == "__main__":
    run_benchmarks()