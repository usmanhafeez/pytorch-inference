
import statistics
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "fashion_cnn.pt"
INPUT_PATH = PROJECT_DIR / "models" / "test_input.bin"

WARMUP_RUNS = 30
BENCHMARK_RUNS = 500

torch.set_num_threads(1)

# Load the same TorchScript model used by C++
model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
model.eval()

# Load the same float32 image used by C++
pixels = np.fromfile(INPUT_PATH, dtype=np.float32)

if pixels.size != 28 * 28:
    raise ValueError(
        f"Expected 784 pixels, got {pixels.size}"
    )

input_tensor = torch.from_numpy(
    pixels.copy().reshape(1, 1, 28, 28)
)

latencies = []

with torch.inference_mode():

    # Warm up before measuring
    for _ in range(WARMUP_RUNS):
        model(input_tensor)

    # Measure inference only
    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        model(input_tensor)

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

print("=== Python TorchScript CPU Benchmark ===")
print("CPU threads:", torch.get_num_threads())
print("Warmup runs:", WARMUP_RUNS)
print("Measured runs:", BENCHMARK_RUNS)

print(f"Mean latency: {statistics.mean(latencies):.4f} ms")
print(f"Median latency: {statistics.median(latencies):.4f} ms")
print(f"Min latency: {min(latencies):.4f} ms")
print(f"Max latency: {max(latencies):.4f} ms")
print(
    f"Estimated throughput: "
    f"{1000 / statistics.mean(latencies):.2f} images/sec"
)

