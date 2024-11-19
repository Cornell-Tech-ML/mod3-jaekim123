"""Timing utilities for benchmarking tensor operations."""

import random
from collections import defaultdict
import time
import sys
import numpy as np
import minitorch

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)


def run_matmul(backend, size: int = 16) -> None:
    """Run matrix multiplication benchmark.

    Performs matrix multiplication on random tensors of specified size.

    Args:
        backend: Tensor backend to use for computations
        size: Size of the matrices to multiply (default: 16)

    Returns:
        None
    """
    batch_size = 2
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        fast_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()
            fast_time = end_fast - start_fast
            fast_times.append(fast_time)

        times[size]["fast"] = np.mean(fast_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")