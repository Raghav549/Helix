"""CPU/GPU runtime selection respecting affinity and Linux CPU quotas."""
from __future__ import annotations
import math
import os


def available_threads() -> int:
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max', encoding='utf-8').read().split()
        if quota != 'max':
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def configure_threads() -> int:
    import torch
    n = int(os.environ.get('HELIX_THREADS', available_threads()))
    if n < 1:
        raise ValueError('HELIX_THREADS must be positive')
    torch.set_num_threads(n)
    return n
