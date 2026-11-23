
import time
import statistics
from pathlib import Path

import torch

from model import FashionCNN

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "fashion_cnn.pth"

WARMUP_RUNS = 30
BENCHMARK_RUNS = 500

model = FashionCNN()
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

input_tensor = torch.randn(1, 1, 28, 28)


def benchmark(num_threads):
    torch.set_num_threads(num_threads)

    latencies = []

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)

        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()
            model(input_tensor)
            elapsed = time.perf_counter() - start

            latencies.append(elapsed * 1000)

    mean = statistics.mean(latencies)
    median = statistics.median(latencies)

    return mean, median


print("=== CPU Thread Benchmark ===")

for threads in [1, 2]:
    mean, median = benchmark(threads)

    print(f"\nCPU threads: {threads}")
    print(f"Mean latency: {mean:.4f} ms")
    print(f"Median latency: {median:.4f} ms")
    print(f"Estimated throughput: {1000 / mean:.2f} images/sec")

