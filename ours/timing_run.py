"""Clean timing of the key variants: one run at a time on an otherwise idle machine (the parallel evaluation runs
share the CPU, so their latencies are only roughly comparable).   python ours/timing_run.py
Writes results/contrib/timing.json (mean / p99 latency per flow, throughput) for the baseline, C1, the single forest and C3."""
import json, sys, time
import numpy as np
sys.path.insert(0, "ours")
import severity_pwpae as sp

ds, seeds = "CICIDS2017", [1]
tuned = json.load(open("results/contrib/tuned.json"))
variants = {
    "Baseline PWPAE": lambda s: sp.run(ds, s),
    "C1 severity-aware": lambda s: sp.run(ds, s, signal="error", tau=tuple(tuned["error_d0"])),
    "C1b always-react + boost": lambda s: sp.run(ds, s, signal="error", level_mode="uniform_high", act_boost=True),
    "single 10-tree forest": lambda s: sp.run_single(ds, s),
    "C3 gated ensemble": lambda s: sp.run_gated(ds, s),
}
out = {}
for name, f in variants.items():
    mean, p99, thr = [], [], []
    for s in seeds:
        lat = f(s)["lat"]
        mean.append(lat.mean() * 1e3); p99.append(np.percentile(lat, 99) * 1e3); thr.append(len(lat) / lat.sum())
    out[name] = dict(lat_ms=float(np.mean(mean)), lat_p99_ms=float(np.mean(p99)), thr=float(np.mean(thr)))
    print(name, out[name], flush=True)
json.dump(out, open("results/contrib/timing.json", "w"), indent=1)
