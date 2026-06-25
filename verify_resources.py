#!/usr/bin/env python3
"""Smoke-test that all gathered resources load and the QML stack works.
Run: source .venv/bin/activate && python verify_resources.py"""
import glob, json, os, sys

ok = True

def check(name, cond):
    global ok
    print(f"[{'OK ' if cond else 'FAIL'}] {name}")
    ok = ok and cond

# 1. papers
pdfs = glob.glob("papers/*.pdf")
check(f"papers: {len(pdfs)} PDFs (>=15)", len(pdfs) >= 15)
check("deep notes json", os.path.exists("paper_search_results/deep_notes.json"))

# 2. frameworks
try:
    import pennylane as qml
    import numpy as np
    dev = qml.device("lightning.qubit", wires=20)
    check("PennyLane lightning.qubit 20 qubits", True)
except Exception as e:
    check(f"PennyLane 20q ({e})", False)
for pkg in ("qiskit", "qiskit_machine_learning", "sklearn"):
    try:
        __import__(pkg); check(f"import {pkg}", True)
    except Exception as e:
        check(f"import {pkg} ({e})", False)

# 3. datasets load + separation
try:
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    d = np.load("datasets/discrete_log/dlp_n20.npz")
    Xb, y = d["x_bits"], d["y"]
    acc = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                          Xb, y, cv=3).mean()
    check(f"DLP n=20 loads; classical RF acc={acc:.3f} (~0.5 expected)", 0.4 < acc < 0.6)

    e = np.load("datasets/engineered_kernel/engk_q20.npz")
    K_Q, K_C, ye = e["K_Q"], e["K_C"], e["y"]
    n = len(ye); tr, te = np.arange(int(0.7*n)), np.arange(int(0.7*n), n)
    qa = SVC(kernel="precomputed").fit(K_Q[np.ix_(tr, tr)], ye[tr]).score(K_Q[np.ix_(te, tr)], ye[te])
    ca = SVC(kernel="precomputed").fit(K_C[np.ix_(tr, tr)], ye[tr]).score(K_C[np.ix_(te, tr)], ye[te])
    check(f"Engineered n=20: quantum acc={qa:.3f} > classical acc={ca:.3f}", qa > ca + 0.1)
except Exception as e:
    check(f"dataset checks ({e})", False)

# 4. docs
for doc in ("literature_review.md", "resources.md", "papers/README.md",
            "datasets/README.md", "code/README.md"):
    check(f"doc {doc}", os.path.exists(doc))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
