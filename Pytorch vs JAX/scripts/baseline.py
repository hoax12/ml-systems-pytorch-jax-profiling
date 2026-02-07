import torch
import numpy as np

# Config
BATCH_SIZES = [8, 16, 32, 64]
SEQ_LEN = 128
NUM_STEPS = 50  # Keep it short for profiling
WARMUP_STEPS = 10
MODEL_ID = "distilbert-base-uncased"

def get_dummy_data_torch(batch_size, seq_len=SEQ_LEN, vocab_size=30522):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64)
    labels = torch.randint(0, 2, (batch_size,))
    return input_ids, attention_mask, labels

def get_dummy_data_numpy(batch_size, seq_len=SEQ_LEN, vocab_size=30522):
    # JAX/Flax expects numpy/jax arrays
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
    labels = np.random.randint(0, 2, (batch_size,))
    return input_ids, attention_mask, labels