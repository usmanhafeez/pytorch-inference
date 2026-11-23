
# PyTorch CNN Inference with C++ LibTorch

A computer vision project demonstrating PyTorch CNN training, TorchScript model export, standalone C++ inference, CPU performance benchmarking, and automated 
testing.

## Project Overview

This project trains a Convolutional Neural Network (CNN) to classify Fashion-MNIST clothing images into 10 categories.

The trained model is exported to TorchScript and executed using the LibTorch C++ API.

The project focuses on:
- Python and C++ integration
- Machine learning inference
- CPU performance analysis
- Numerical correctness
- Automated testing

## Technologies

- Python 3.11
- PyTorch 2.2.2
- torchvision 0.17.2
- NumPy 1.26.4
- C++17
- LibTorch
- CMake
- pytest

## Model Architecture

Input: 28 x 28 grayscale image

1. Conv2d (1 to 8 channels) + ReLU
2. MaxPool2d (2 x 2)
3. Conv2d (8 to 16 channels) + ReLU
4. MaxPool2d (2 x 2)
5. Flatten (784 features)
6. Linear (784 to 64) + ReLU
7. Linear (64 to 10)

Training configuration:
- Dataset: Fashion-MNIST
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 2
- Loss function: CrossEntropyLoss

**Test accuracy: 86.19%**

## Project Structure

```text
arm-pytorch-inference/
├── CMakeLists.txt
├── requirements.txt
├── README.md
├── cpp/
│   ├── main.cpp
│   ├── inference.cpp
│   └── benchmark.cpp
├── src/
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── export_model.py
│   ├── export_test_input.py
│   ├── benchmark.py
│   ├── benchmark_torchscript.py
│   ├── tensor_basics.py
│   └── simple_neural_network.py
└── tests/
    ├── test_model.py
    └── test_cpp_inference.py
```

## Setup

### 1. Create a virtual environment

Python 3.11 is recommended for the original environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

PyTorch package availability depends on the operating system, architecture, and Python version.

### 3. Train the CNN

```bash
mkdir -p models
python src/train.py
```

This downloads Fashion-MNIST and saves the trained model weights.

### 4. Export the model and test image

```bash
python src/export_model.py
python src/export_test_input.py
```

### 5. Build C++ applications

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake --build build -j1
```

### 6. Run inference

Python:

```bash
python src/inference.py
```

C++:

```bash
./build/torch_cpp_inference
```

### 7. Run benchmarks

Python TorchScript:

```bash
python src/benchmark_torchscript.py
```

C++ LibTorch:

```bash
./build/torch_cpp_benchmark
```

### 8. Run automated tests

```bash
python -m pytest tests/ -v
```

Build the C++ inference executable before running the integration test.

## CPU Benchmark Results

Hardware: Intel MacBook Pro (2012), macOS Ventura.

Benchmark configuration:
- Same TorchScript model and test input
- CPU threads: 1
- Batch size: 1
- Warm-up iterations: 30
- Measured iterations: 500
- Three runs per implementation

### Median Inference Latency

| Run | Python TorchScript | C++ LibTorch |
|---|---:|---:|
| 1 | 0.2301 ms | 0.2068 ms |
| 2 | 0.2214 ms | 0.2079 ms |
| 3 | 0.2231 ms | 0.2081 ms |
| **Median of runs** | **0.2231 ms** | **0.2079 ms** |

C++ achieved approximately **6.8% lower median inference latency** under the tested conditions.

Results are specific to this hardware, workload, and benchmark methodology.

### CPU Thread Comparison

| CPU threads | Mean latency | Median latency |
|---|---:|---:|
| 1 | 0.3717 ms | 0.3212 ms |
| 2 | 0.6316 ms | 0.3993 ms |

Single-threaded execution was faster for this small CNN in the measured experiment.

## Automated Tests

The project includes four pytest tests:

1. CNN output shape validation
2. Trained weights loading
3. Python and TorchScript numerical equivalence
4. C++ inference integration

Latest result:

**4 passed in 6.45 seconds**

## Key Learnings

- CNN training and inference with PyTorch
- TorchScript model serialization
- Standalone C++ inference using LibTorch
- CMake compilation and native library linking
- CPU threading and inference latency
- Benchmark methodology and measurement variability
- Automated validation of ML inference

## Future Improvements

- Numerical equivalence testing between Python and C++ outputs
- p95 and p99 latency measurements
- Additional batch-size benchmarks
- GitHub Actions CI
- Evaluation on Arm-based CPU hardware

## Scope

The current benchmarks were performed on an Intel CPU, not Arm hardware.

