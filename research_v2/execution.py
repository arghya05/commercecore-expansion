"""Execution policy: research workloads run on RunPod, never on the laptop."""
import os
import platform


def require_runpod():
    if platform.system() != 'Linux' or not os.environ.get('RUNPOD_POD_ID'):
        raise RuntimeError(
            'Research execution is restricted to RunPod. Do not run tests, model '
            'inference, data builds, training, or PDF compilation on the laptop.'
        )

