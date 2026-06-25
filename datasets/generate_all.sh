#!/usr/bin/env bash
# Regenerate all datasets + sanity checks. Run from workspace root.
set -e
source .venv/bin/activate

echo "=== Discrete-log (DLP) datasets ==="
for n in 12 16 20; do
  python datasets/generators/discrete_log_dataset.py --n-bits $n --n-samples 2000 --seed 0
done

echo "=== Engineered projected-kernel datasets ==="
for q in 8 10 20; do
  python datasets/generators/engineered_kernel_dataset.py --n-qubits $q --n-samples 250 --gamma 0.1 --sanity --seed 0
done
echo "Done. See datasets/*/*_meta.json for sanity checks."
