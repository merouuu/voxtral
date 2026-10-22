"""Linear probes (L2 logistic regression) per hook and task, actor-disjoint split, normalized balanced accuracy.

Usage: .venv/bin/python probe.py outputs [n_seeds]
"""
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import split_by_actor

TASKS = {"text": "sentence", "gender": "gender", "emotion": "emotion"}

out = Path(sys.argv[1])
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
feats = np.load(out / "features.npz")
meta = list(csv.DictReader(open(out / "meta.csv")))
actors = np.array([int(m["actor"]) for m in meta])

rows = []
for seed in range(n_seeds):
    train, test = split_by_actor([{"actor": a, "i": i} for i, a in enumerate(actors)], seed=seed)
    tr, te = [c["i"] for c in train], [c["i"] for c in test]
    for hook in feats.files:
        if hook.endswith("_all") and seed > 0:  # Q8 ablation: one seed is enough
            continue
        X = feats[hook].astype(np.float32)
        for task, col in TASKS.items():
            y = np.array([m[col] for m in meta])
            grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                {"logisticregression__C": [0.01, 0.1, 1, 10]},
                                cv=GroupKFold(5), scoring="balanced_accuracy", n_jobs=8)
            grid.fit(X[tr], y[tr], groups=actors[tr])
            acc = balanced_accuracy_score(y[te], grid.predict(X[te]))
            chance = 1 / len(set(y))
            rows.append((hook, task, seed, acc, (acc - chance) / (1 - chance)))
            print(*rows[-1], grid.best_params_, flush=True)

with open(out / "probe_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["hook", "task", "n_seeds", "acc", "acc_norm", "std"])
    for hook in feats.files:
        for task in TASKS:
            r = np.array([(a, n) for h, t, _, a, n in rows if h == hook and t == task])
            w.writerow([hook, task, len(r), *r.mean(0).round(4), r[:, 1].std().round(4)])
print(open(out / "probe_results.csv").read())
