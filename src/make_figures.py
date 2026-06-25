"""Generate figures from results/exp_a.json and results/exp_b.json."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
A = json.load(open("results/exp_a.json"))
B = json.load(open("results/exp_b.json"))


def mean_std(entry, model):
    m = entry["models"].get(model)
    if m is None:
        return None, None
    return m["test"]["mean"], m["test"]["std"]


# --- Fig 1: headline bar chart at n=20 (seed 0), all models -----------------
e = A["n20_s0"]
order = ["Quantum: Projected kernel", "Quantum: Fidelity kernel",
         "Classical: RBF-SVM (tuned)", "Classical: RBF kernel (precomp.)",
         "Classical: Gradient Boosting", "Classical: Random Forest",
         "Classical: MLP", "Classical: Linear-SVM",
         "Classical: Logistic Reg.", "Baseline: Majority class"]
order = [m for m in order if m in e["models"]]
means = [e["models"][m]["test"]["mean"] for m in order]
stds = [e["models"][m]["test"]["std"] for m in order]
colors = ["#1b7837" if m.startswith("Quantum: Projected") else
          "#7fbf7b" if m.startswith("Quantum") else
          "#999999" if m.startswith("Baseline") else "#998ec3" for m in order]
labels = [m.replace("Classical: ", "").replace("Quantum: ", "Q: ")
          .replace("Baseline: ", "") for m in order]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(range(len(order)), means, xerr=stds, color=colors,
               edgecolor="black", capsize=3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.axvline(0.5, ls="--", c="red", label="chance (0.50)")
ax.set_xlabel("Test accuracy (mean ± std over 10 splits)")
ax.set_xlim(0, 1)
ax.set_title(f"Engineered task, n=20 qubits  (geometric difference g={e['g']:.1f})")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.01, i, f"{m:.2f}", va="center", fontsize=8)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_engineered_n20_bars.png", dpi=150)
plt.close()

# --- Fig 2: accuracy vs qubit count (projected Q vs best classical) ---------
ns = [8, 10, 20]
qproj, qproj_e, cbest, cbest_e, fid, gvals = [], [], [], [], [], []
for n in ns:
    seeds = [k for k in A if k.startswith(f"n{n}_")]
    qm = [A[k]["models"]["Quantum: Projected kernel"]["test"]["mean"] for k in seeds]
    qproj.append(np.mean(qm)); qproj_e.append(np.std(qm))
    cb = []
    for k in seeds:
        cls = {m: v for m, v in A[k]["models"].items()
               if m.startswith("Classical") or m.startswith("Baseline")}
        cb.append(max(v["test"]["mean"] for v in cls.values()))
    cbest.append(np.mean(cb)); cbest_e.append(np.std(cb))
    gvals.append(np.mean([A[k]["g"] for k in seeds]))
    f0 = A[f"n{n}_s0"]["models"].get("Quantum: Fidelity kernel")
    fid.append(f0["test"]["mean"] if f0 else np.nan)
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(ns, qproj, yerr=qproj_e, marker="o", lw=2, capsize=4,
            color="#1b7837", label="Q: Projected kernel")
ax.plot(ns, fid, marker="s", ls="--", color="#7fbf7b", label="Q: Fidelity kernel (seed 0)")
ax.errorbar(ns, cbest, yerr=cbest_e, marker="^", lw=2, capsize=4,
            color="#998ec3", label="Best classical model")
ax.axhline(0.5, ls="--", c="red", label="chance")
for n, q, c in zip(ns, qproj, cbest):
    ax.annotate(f"Δ={q-c:.2f}", (n, (q + c) / 2), fontsize=8, ha="center")
ax.set_xticks(ns)
ax.set_xlabel("number of qubits n")
ax.set_ylabel("test accuracy")
ax.set_ylim(0.3, 1)
ax.set_title("Quantum advantage vs qubit count (engineered task)")
ax2 = ax.twinx()
ax2.plot(ns, gvals, marker="x", color="gray", alpha=0.6)
ax2.set_ylabel("geometric difference g", color="gray")
ax.legend(loc="center left")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_accuracy_vs_qubits.png", dpi=150)
plt.close()

# --- Fig 3: discrete-log — classical at chance vs quantum across n ----------
ns_b = [int(k[1:]) for k in sorted(B, key=lambda x: int(x[1:]))]
qb, cbest_b = [], []
classical_all = {}
for k in sorted(B, key=lambda x: int(x[1:])):
    qb.append(B[k]["models"]["Quantum: Interval-state kernel"]["test"]["mean"])
    for m, v in B[k]["models"].items():
        if m.startswith("Quantum"):
            continue
        classical_all.setdefault(m, []).append(v["test"]["mean"])
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(ns_b, qb, marker="o", lw=2.5, color="#1b7837",
        label="Q: Interval-state kernel")
for m, vals in classical_all.items():
    ax.plot(ns_b, vals, marker=".", alpha=0.7,
            label=m.replace("Classical: ", "").replace("Baseline: ", ""))
ax.axhline(0.5, ls="--", c="red", lw=1)
ax.set_xticks(ns_b)
ax.set_xlabel("discrete-log problem size n (bits ≈ qubits)")
ax.set_ylabel("test accuracy")
ax.set_ylim(0.3, 1.05)
ax.set_title("Discrete-log: off-the-shelf classical at chance, quantum separates")
ax.legend(fontsize=7, ncol=2, loc="center right")
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_discrete_log.png", dpi=150)
plt.close()

# --- Fig 4: generalization gap (engineered, n=20) ---------------------------
e = A["n20_s0"]
gap_models = [m for m in order if m in e["models"]]
train_m = [e["models"][m]["train"]["mean"] for m in gap_models]
test_m = [e["models"][m]["test"]["mean"] for m in gap_models]
x = np.arange(len(gap_models))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, train_m, 0.4, label="train", color="#cccccc", edgecolor="k")
ax.bar(x + 0.2, test_m, 0.4, label="test", color="#1b7837", edgecolor="k")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Classical: ", "").replace("Quantum: ", "Q: ")
                    .replace("Baseline: ", "") for m in gap_models],
                   rotation=40, ha="right", fontsize=8)
ax.axhline(0.5, ls="--", c="red")
ax.set_ylabel("accuracy")
ax.set_title("Train vs test accuracy, n=20 (overfit = high train, low test)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_generalization_gap.png", dpi=150)
plt.close()

print("figures written:", os.listdir(FIG))
