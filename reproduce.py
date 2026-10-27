"""Run the unmodified basic_v2 paper protocol and verify numerical outputs."""
import os
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parent
env = os.environ.copy()
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    env[key] = "1"
for script in ("Experiments/run_basic_experiments.py", "Experiments/summarize_basic.py", "verify_results.py"):
    subprocess.run([sys.executable, str(root / script)], cwd=root, env=env, check=True)
