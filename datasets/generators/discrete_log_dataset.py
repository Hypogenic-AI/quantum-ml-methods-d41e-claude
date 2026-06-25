#!/usr/bin/env python3
"""
Discrete-Logarithm (DLP) dataset generator.

Reproduces the provable-quantum-advantage learning problem of
Liu, Arunachalam & Temme, "A rigorous and robust quantum speed-up in
supervised machine learning", Nature Physics 17, 1013-1017 (2021),
arXiv:2010.02174.

Construction
------------
Let p be a large prime and g a generator of the cyclic multiplicative
group Z_p^* (order p-1). For input x in Z_p^* define its discrete log
  L(x) = log_g(x)   (the unique e in {0,...,p-2} with g^e = x mod p).
Pick a secret s. The binary label is
  y(x) = +1  if  (L(x) - s) mod (p-1)  in  [0, (p-1)/2)
  y(x) = -1  otherwise.

The labelling is a half-space in *log space*. A quantum computer can
build a feature map that linearly separates the two classes (using the
group structure / period finding), giving an SVM that classifies with
high accuracy. Any *classical* learner with only black-box access to x
must effectively compute discrete logs, which is believed hard
(sub-exponential at best); Liu et al. prove no classical learner beats
random guessing by more than inverse-polynomial, assuming DLP hardness.

This is the cleanest "classical struggles, quantum excels" task and is
the headline target for the research hypothesis. The number of qubits
needed to operate on Z_p^* is n = ceil(log2(p)), so choosing p near
2^20 gives a ~20-qubit problem suitable for simulation.

Outputs
-------
For a chosen bit-size n the generator writes:
  X        : (N,) integer array of group elements (and an (N,n) bit view)
  y        : (N,) labels in {-1,+1}
  metadata : p, g, s, n, class balance

Note: for honest evaluation the *labels* are computed here with full
knowledge of the discrete log (we know s and can build the full log
table for simulable p). A learner is given only X (as integers / bits)
and y_train, never s or g's log table.
"""
import argparse
import json
import os
import numpy as np


def is_prime(n):
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_safe_prime_near(target):
    """Find a prime p with a known small generator near `target`.
    We look for a 'safe prime' p = 2q+1 (q prime) so that 2 (or a small
    int) is a generator, which makes building the log table trivial."""
    base = target | 1  # make odd
    for delta in range(0, 10_000_000, 2):
        for cand in (base + delta, base - delta):
            if cand < 5 or cand % 2 == 0 or not is_prime(cand):
                continue
            q = (cand - 1) // 2
            if is_prime(q):
                # find generator: for safe prime, g is a generator iff
                # g^2 != 1 and g^q != 1 mod p.
                for g in range(2, 100):
                    if pow(g, 2, cand) != 1 and pow(g, q, cand) != 1:
                        return cand, g
    raise RuntimeError("no safe prime found")


def build_log_table(p, g):
    """Full discrete-log table via baby-step (feasible for simulable p)."""
    log = np.full(p, -1, dtype=np.int64)
    cur = 1
    for e in range(p - 1):
        log[cur] = e
        cur = cur * g % p
    return log


def generate(n_bits=20, n_samples=2000, seed=0, balance=True):
    rng = np.random.default_rng(seed)
    target = 1 << n_bits
    p, g = find_safe_prime_near(target)
    n = (p - 1).bit_length()  # qubits to address Z_{p-1}
    log = build_log_table(p, g)
    half = (p - 1) // 2
    s = int(rng.integers(0, p - 1))

    # sample distinct non-zero group elements
    xs = rng.choice(np.arange(1, p), size=min(n_samples, p - 1), replace=False)
    L = log[xs]
    y = np.where(((L - s) % (p - 1)) < half, 1, -1).astype(np.int8)

    if balance:
        pos = np.where(y == 1)[0]
        neg = np.where(y == -1)[0]
        m = min(len(pos), len(neg), n_samples // 2)
        idx = np.concatenate([rng.choice(pos, m, replace=False),
                              rng.choice(neg, m, replace=False)])
        rng.shuffle(idx)
        xs, y = xs[idx], y[idx]

    # bit-vector view of each integer (n bits, MSB first) -> model input
    bits = ((xs[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.float32)
    meta = dict(p=int(p), g=int(g), secret_s=int(s), n_bits=int(n),
                n_qubits_required=int(n), n_samples=int(len(xs)),
                class_balance=float(np.mean(y == 1)),
                reference="Liu, Arunachalam, Temme 2021 (arXiv:2010.02174)")
    return xs.astype(np.int64), bits, y, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bits", type=int, default=20,
                    help="approx log2(p); sets #qubits ~= n-bits")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="datasets/discrete_log")
    args = ap.parse_args()

    xs, bits, y, meta = generate(args.n_bits, args.n_samples, args.seed)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f"dlp_n{args.n_bits}.npz"),
                        x_int=xs, x_bits=bits, y=y)
    with open(os.path.join(args.out, f"dlp_n{args.n_bits}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # small human-readable sample
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    with open(os.path.join(args.out, "samples", f"sample_n{args.n_bits}.json"), "w") as f:
        json.dump([{"x": int(xs[i]), "y": int(y[i])} for i in range(min(10, len(xs)))],
                  f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"Saved {len(xs)} samples to {args.out}")


if __name__ == "__main__":
    main()
