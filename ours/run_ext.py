"""Extension experiments on top of the PWPAE/MSANA baselines (River 0.8.0, py3.8).

Three jobs, all on the sampled IoTID20 / CICIDS2017 streams shipped with the PWPAE repo,
10% train / 90% prequential test, no shuffling (same protocol as the authors' notebooks):
  base   : run one of the 4 PWPAE base learners, cache p(class=1) + predictions + latency
  sample : ARF-ADWIN behind different resampling strategies
  (ensemble weighting + detectors are computed offline from the cache in analyse.py)
Usage: python experiments/run_ext.py   (run from the sda/ folder)
"""
import json, math, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import numpy as np
import pandas as pd
from river import ensemble
from river.drift import ADWIN, DDM
from river import imblearn

OUT = "results/ext"
DATA = {
    "IoTID20": ("baseline/data/IoT_2020_b_0.01_fs.csv", "Label", [270, 600]),
    "CICIDS2017": ("baseline/data/cic_0.01km.csv", "Labelb", [9900, 12600, 14200, 15800, 18800, 21600]),
}
SEEDS = [1, 2, 3]
BASE = ["ARF-ADWIN", "SRP-ADWIN", "ARF-DDM", "SRP-DDM"]


def load(name):
    path, lab, dr = DATA[name]
    df = pd.read_csv(path)
    y = df[lab].astype(int).values
    recs = df.drop(columns=[lab]).to_dict("records")
    n0 = int(0.1 * len(df))
    return recs[:n0], y[:n0], recs[n0:], y[n0:], dr


def make_base(kind, seed):
    if kind == "ARF-ADWIN":
        return ensemble.AdaptiveRandomForestClassifier(n_models=3, drift_detector=ADWIN(), seed=seed)
    if kind == "SRP-ADWIN":
        return ensemble.SRPClassifier(n_models=3, drift_detector=ADWIN(), seed=seed)
    if kind == "ARF-DDM":
        return ensemble.AdaptiveRandomForestClassifier(n_models=3, drift_detector=DDM(), warning_detector=DDM(), seed=seed)
    if kind == "SRP-DDM":
        return ensemble.SRPClassifier(n_models=3, drift_detector=DDM(), warning_detector=DDM(), seed=seed)
    raise ValueError(kind)


class Resampler:
    """Online resampling wrappers: oob / uob (Wang et al.), sw (sliding-window class prior), smote."""

    def __init__(self, model, mode, seed, theta=0.9, window=1000, cap=10.0):
        self.m, self.mode, self.theta, self.cap = model, mode, theta, cap
        self.rng = np.random.RandomState(seed)
        self.w = {0: 0.5, 1: 0.5}
        self.win = deque(maxlen=window)
        self.buf = deque(maxlen=200)  # minority buffer for smote

    def predict_proba_one(self, x):
        return self.m.predict_proba_one(x)

    def _class_share(self, y):
        if self.mode == "sw":
            self.win.append(y)
            n1 = sum(self.win)
            n = len(self.win)
            return {0: max(n - n1, 1) / n, 1: max(n1, 1) / n}
        for c in (0, 1):
            self.w[c] = self.theta * self.w[c] + (1 - self.theta) * (1.0 if y == c else 0.0)
        return self.w

    def learn_one(self, x, y):
        if self.mode == "smote":
            self.win.append(y)
            n1 = sum(self.win)
            minority = 1 if n1 < len(self.win) / 2 else 0
            self.m.learn_one(x, y)
            if y == minority:
                if len(self.buf) and self.buf.maxlen:
                    nb = self.buf[self.rng.randint(len(self.buf))]
                    a = self.rng.rand()
                    self.m.learn_one({k: x[k] + a * (nb[k] - x[k]) for k in x}, y)
                self.buf.append(x)
            return
        w = self._class_share(y)
        mn, mx = min(w.values()), max(w.values())
        if self.mode in ("oob", "sw"):
            lam = min(mx / max(w[y], 1e-6), self.cap)
        else:  # uob
            lam = mn / max(w[y], 1e-6)
        for _ in range(self.rng.poisson(lam)):
            self.m.learn_one(x, y)


def make_sampler(kind, seed):
    base = make_base("ARF-ADWIN", seed)
    if kind == "none":
        return base
    if kind == "RUS-40/60":
        return imblearn.RandomUnderSampler(base, desired_dist={0: 0.4, 1: 0.6}, seed=seed)
    if kind == "ROS-40/60":
        return imblearn.RandomOverSampler(base, desired_dist={0: 0.4, 1: 0.6}, seed=seed)
    if kind == "RUS-50/50":
        return imblearn.RandomUnderSampler(base, desired_dist={0: 0.5, 1: 0.5}, seed=seed)
    if kind == "ROS-50/50":
        return imblearn.RandomOverSampler(base, desired_dist={0: 0.5, 1: 0.5}, seed=seed)
    return Resampler(base, kind, seed)  # oob uob sw smote


def prequential(model, tr, ytr, te, yte):
    for x, y in zip(tr, ytr):
        model.learn_one(x, int(y))
    n = len(te)
    p1, pred, lat = np.zeros(n), np.zeros(n, dtype=int), np.zeros(n)
    pc = time.perf_counter
    for i, (x, y) in enumerate(zip(te, yte)):
        t0 = pc()
        pr = model.predict_proba_one(x)
        p1[i] = pr.get(1, 0.0)
        pred[i] = max(pr, key=pr.get) if pr else 0
        model.learn_one(x, int(y))
        lat[i] = pc() - t0
    return p1, pred, lat


def job(args):
    kind, ds, seed, name = args
    out = f"{OUT}/{kind}__{ds}__{name.replace('/', '_')}__s{seed}.npz"
    if os.path.exists(out):
        return out
    tr, ytr, te, yte, _ = load(ds)
    model = make_base(name, seed) if kind == "base" else make_sampler(name, seed)
    p1, pred, lat = prequential(model, tr, ytr, te, yte)
    try:
        size = len(pickle.dumps(model))
    except Exception:
        size = -1
    np.savez(out, p1=p1, pred=pred, lat=lat, y=yte, size=size)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for ds in DATA:
        for s in SEEDS:
            jobs += [("base", ds, s, b) for b in BASE]
            jobs += [("sample", ds, s, k) for k in
                     ["none", "RUS-40/60", "ROS-40/60", "RUS-50/50", "ROS-50/50", "oob", "uob", "sw", "smote"]]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {r}  ({time.time() - t0:.0f}s)", flush=True)
