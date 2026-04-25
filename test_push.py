#!/usr/bin/env python3
"""Test script to push a single experiment to Kaggle."""

import os
import json
import time
import subprocess
import shutil
from pathlib import Path

os.environ['NUM_LAYERS'] = '11'
os.environ['MODEL_DIM'] = '512'
os.environ['VOCAB_SIZE'] = '8192'
os.environ['QK_GAIN_INIT'] = '5.25'
os.environ['NUM_LOOPS'] = '2'
os.environ['RUN_ID'] = 'arch-test-1'

target = '/tmp/pg-arch-test'
os.makedirs(target, exist_ok=True)

# Create wrapper
env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(os.environ.items()) if k in ['NUM_LAYERS', 'MODEL_DIM', 'VOCAB_SIZE', 'QK_GAIN_INIT', 'NUM_LOOPS', 'RUN_ID'])
train_source = open('train_kaggle.py').read()
wrapper = target + '/run_kernel.py'
with open(wrapper, 'w') as f:
    f.write(f"import os\n{env_lines}\n{train_source}\n")

# Metadata
kernel_id = f'dhruv2mars/pg-arch-test-{int(time.time())}'
metadata = {
    'id': kernel_id,
    'title': 'PG Arch Test 1',
    'code_file': 'run_kernel.py',
    'language': 'python',
    'kernel_type': 'script',
    'is_private': False,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': True,
    'dataset_sources': [],
    'competition_sources': [],
    'kernel_sources': [],
    'model_sources': [],
}
with open(target + '/kernel-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Pushing: {kernel_id}')
result = subprocess.run(['kaggle', 'kernels', 'push', '-p', target, '--accelerator', 'NvidiaTeslaT4'], capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)