"""Shared utilities: seeding, environment logging, classical baselines,
repeated-split evaluation, geometric difference, statistical tests."""
import json
import os
import random
import sys
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from scipy import stats

SEED = 42


def ensure_root_cwd():
    """chdir to repo root (parent of src/) so relative paths are stable."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    return root


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_env(path="results/environment.json"):
    import sklearn
    import scipy
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import pennylane as qml
        info["pennylane"] = qml.__version__
    except Exception:
        info["pennylane"] = None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


# ---------------------------------------------------------------------------
# Geometric difference g(K_C || K_Q)  (Huang et al. 2021) — pre-training
# diagnostic that upper-bounds achievable quantum advantage.
# ---------------------------------------------------------------------------
def geometric_difference(K_C, K_Q, lam=1e-3):
    n = K_C.shape[0]
    wq, vq = np.linalg.eigh(K_Q)
    wq = np.clip(wq, 0, None)
    sqrtKQ = (vq * np.sqrt(wq)) @ vq.T
    inv = np.linalg.inv(K_C + lam * np.eye(n))
    M = sqrtKQ @ inv @ sqrtKQ
    M = (M + M.T) / 2
    w = np.linalg.eigvalsh(M)
    return float(np.sqrt(max(w[-1], 0.0)))


# ---------------------------------------------------------------------------
# Classical baselines on raw feature vectors X.
# ---------------------------------------------------------------------------
def classical_models_on_features(seed):
    """Return dict name -> (estimator, needs_scaling). RBF-SVM is CV-tuned."""
    rbf = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]},
        cv=3, n_jobs=-1,
    )
    return {
        "Classical: RBF-SVM (tuned)": (rbf, True),
        "Classical: Linear-SVM": (SVC(kernel="linear", C=1.0), True),
        "Classical: Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1), False),
        "Classical: Gradient Boosting": (
            GradientBoostingClassifier(random_state=seed), False),
        "Classical: MLP": (
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                          random_state=seed), True),
        "Classical: Logistic Reg.": (
            LogisticRegression(max_iter=2000, random_state=seed), True),
        "Baseline: Majority class": (
            DummyClassifier(strategy="most_frequent"), False),
    }


def eval_feature_models(X, y, n_splits=10, test_size=0.3, seed=SEED):
    """Repeated stratified-split evaluation of classical feature models.
    Returns dict name -> {'test': [...], 'train': [...]} per split."""
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                 random_state=seed)
    results = {}
    splits = list(sss.split(X, y))
    for tr, te in splits:
        models = classical_models_on_features(seed)
        for name, (est, scale) in models.items():
            Xt, Xe = X[tr], X[te]
            if scale:
                sc = StandardScaler().fit(Xt)
                Xt, Xe = sc.transform(Xt), sc.transform(Xe)
            est.fit(Xt, y[tr])
            tr_acc = est.score(Xt, y[tr])
            te_acc = est.score(Xe, y[te])
            d = results.setdefault(name, {"test": [], "train": []})
            d["test"].append(te_acc)
            d["train"].append(tr_acc)
    return results, splits


def eval_precomputed_kernel(K, y, splits, C=1.0):
    """Evaluate an SVM on a precomputed kernel over the SAME splits used for
    feature models (so comparisons are paired)."""
    out = {"test": [], "train": []}
    for tr, te in splits:
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        out["train"].append(clf.score(K[np.ix_(tr, tr)], y[tr]))
        out["test"].append(clf.score(K[np.ix_(te, tr)], y[te]))
    return out


def summarize(scores):
    a = np.asarray(scores)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}


def paired_test(a, b):
    """Paired t-test + Cohen's d (paired) for a vs b (same splits)."""
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    return {"t": float(t), "p": float(p), "cohens_d": float(d),
            "mean_diff": float(diff.mean())}
