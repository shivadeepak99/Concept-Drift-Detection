"""Tune the severity thresholds on a validation stretch, then run every contribution variant.

  python ours/run_contributions.py tune [workers]   # validation = first 30 % of the stream, seeds 11-13 (never the test seeds)
  python ours/run_contributions.py eval [workers]   # final runs, scored on the last 70 % of the stream
Results go to results/contrib/ (resumable: finished runs are skipped).
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, "ours")
import severity_pwpae as sp

OUT = "results/contrib"
VAL_FRAC = 0.30
GRID = {"error": [(1, 3), (1, 6), (2, 6), (0.5, 2)], "uncertainty": [(0.5, 1.5), (0.5, 3), (1, 3), (1, 6)]}
VAL_SEEDS = [11, 12, 13]
EVAL_SEEDS = {"CICIDS2017": [1, 2, 3, 4, 5], "IoTID20": [1, 2, 3]}
N_TEST = {"CICIDS2017": 25473, "IoTID20": 5627}


def job(spec):
    ds, name, seed, kind, kw, n_stop, folder = spec
    out = f"{OUT}/{folder}/{ds}__{name}__s{seed}.npz"
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if kind == "run":
        r = sp.run(ds, seed, n_stop=n_stop, **kw)
    elif kind == "single":
        r = sp.run_single(ds, seed, n_stop=n_stop, **kw)
    else:
        r = sp.run_gated(ds, seed, n_stop=n_stop, **kw)
    np.savez(out, p1=r["p1"], pred=r["pred"], lat=r["lat"], y=r["y"], active=r.get("active", np.zeros(1)),
             events=json.dumps(r["events"]), replaced=r["replaced"])
    return out


def pool(specs, workers):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, s) for s in specs]
        for i, f in enumerate(futs):
            print(f"[{i + 1}/{len(specs)}] {f.result()} ({time.time() - t0:.0f}s)", flush=True)


def tune(workers):
    ds, nstop = "CICIDS2017", int(VAL_FRAC * N_TEST["CICIDS2017"])
    specs = []
    for D in (0, 200):
        for s in VAL_SEEDS:
            specs.append((ds, f"B0_d{D}", s, "run", dict(delay=D), nstop, "val"))
        for sig in GRID:
            for t in GRID[sig]:
                for s in VAL_SEEDS:
                    specs.append((ds, f"{sig}_d{D}_{t[0]}-{t[1]}", s, "run", dict(signal=sig, delay=D, tau=t), nstop, "val"))
    pool(specs, workers)
    from analyse import full_metrics
    import run_ext as rx
    tr, ytr, te, yte, dr = rx.load(ds)
    best, table = {}, []
    for D in (0, 200):
        b0 = np.mean([full_metrics(*(lambda z: (z["y"], z["pred"], z["p1"], z["lat"], [], int(ytr[-1])))(np.load(f"{OUT}/val/{ds}__B0_d{D}__s{s}.npz")))["mcc"] for s in VAL_SEEDS])
        table.append(f"delay {D}: baseline MCC {b0:.4f}")
        for sig in GRID:
            sc = {}
            for t in GRID[sig]:
                v = []
                for s in VAL_SEEDS:
                    z = np.load(f"{OUT}/val/{ds}__{sig}_d{D}_{t[0]}-{t[1]}__s{s}.npz")
                    v.append(full_metrics(z["y"], z["pred"], z["p1"], z["lat"], [], int(ytr[-1]))["mcc"])
                sc[t] = float(np.mean(v)); table.append(f"  {sig} tau={t}: validation MCC {sc[t]:.4f}")
            best[f"{sig}_d{D}"] = max(sc, key=sc.get)
    json.dump({k: list(v) for k, v in best.items()}, open(f"{OUT}/tuned.json", "w"))
    open(f"{OUT}/tuning.txt", "w").write("\n".join(table) + "\nchosen: " + json.dumps({k: list(v) for k, v in best.items()}) + "\n")
    print("\n".join(table)); print("chosen", best)


def evaluate(workers):
    tuned = {k: tuple(v) for k, v in json.load(open(f"{OUT}/tuned.json")).items()}
    specs = []
    for ds, seeds in EVAL_SEEDS.items():
        for s in seeds:
            V = [("B0", "run", {}), ("C1", "run", dict(signal="error", tau=tuned["error_d0"])),
                 ("C2", "run", dict(signal="uncertainty", tau=tuned["uncertainty_d0"])),
                 ("S10", "single", {}), ("G", "gated", {})]
            if ds == "CICIDS2017":
                V += [("C1_uniform_medium", "run", dict(signal="error", level_mode="uniform_medium")),
                      ("C1_uniform_high", "run", dict(signal="error", level_mode="uniform_high")),
                      ("C1_weights_only", "run", dict(signal="error", tau=tuned["error_d0"], act_replay=False, act_replace=False)),
                      ("C1_weights_replay", "run", dict(signal="error", tau=tuned["error_d0"], act_replace=False)),
                      ("B0_d200", "run", dict(delay=200)),
                      ("C1_d200", "run", dict(signal="error", delay=200, tau=tuned["error_d200"])),
                      ("C2_d200", "run", dict(signal="uncertainty", delay=200, tau=tuned["uncertainty_d200"]))]
            for name, kind, kw in V:
                specs.append((ds, name, s, kind, kw, None, "eval"))
    pool(specs, workers)


if __name__ == "__main__":
    stage, workers = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
    {"tune": tune, "eval": evaluate}[stage](workers)
