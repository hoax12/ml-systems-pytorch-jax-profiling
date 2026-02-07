import jax
import jax.numpy as jnp
from flax.training import train_state
from transformers import FlaxDistilBertForSequenceClassification
import optax
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.baseline import BATCH_SIZES, NUM_STEPS, WARMUP_STEPS, get_dummy_data_numpy, MODEL_ID
from profiling.profiler_utils import BenchmarkTracker

def create_train_state(model, learning_rate):
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx)

@jax.jit
def train_step(state, input_ids, attention_mask, labels):
    def loss_fn(params):
        outputs = state.apply_fn(input_ids=input_ids, attention_mask=attention_mask, params=params)
        loss = optax.softmax_cross_entropy_with_integer_labels(outputs.logits, labels).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def run_benchmarks():
    tracker = BenchmarkTracker()
    print(f"JAX Devices: {jax.devices()}")

    model = FlaxDistilBertForSequenceClassification.from_pretrained(MODEL_ID)
    
    for bs in BATCH_SIZES:
        input_ids, attn_mask, labels = get_dummy_data_numpy(bs)
        state = create_train_state(model, 5e-5)
        
        # Warmup (compiles the graph)
        print(f"Compiling JAX (Batch {bs})...")
        for _ in range(WARMUP_STEPS):
            state, _ = train_step(state, input_ids, attn_mask, labels)
            
        # Benchmark
        # Block until ready is crucial for JAX async dispatch
        jax.block_until_ready(state.params) 
        start = time.perf_counter()
        
        for _ in range(NUM_STEPS):
            state, _ = train_step(state, input_ids, attn_mask, labels)
            
        jax.block_until_ready(state.params)
        end = time.perf_counter()
        
        avg_step_time = ((end - start) / NUM_STEPS) * 1000
        
        # JAX Memory Profiling (GPU only)
        try:
            mem_stats = jax.devices()[0].memory_stats()
            peak_mem = mem_stats['peak_bytes_in_use'] / (1024**2)
        except:
            peak_mem = 0 # CPU or unsupported backend
            
        tracker.record("JAX", "JIT_XLA", bs, avg_step_time, peak_mem)
        print(f"Batch {bs}: {avg_step_time:.2f}ms/step, {peak_mem:.0f}MB")

    tracker.save_csv("results/jax_benchmark_data.csv")
    tracker.plot_throughput("results/jax_throughput.png")

if __name__ == "__main__":
    run_benchmarks()