"""Seeds 1-10 for the core comparison on CICIDS2017: baseline vs C1 (severity-scaled) vs "always high", plus the learning-boost reaction C1b.
   python ours/more_seeds.py run [workers]   |   python ours/more_seeds.py report
Pools seeds 1-10 and prints paired differences to the baseline (mean, 95% CI, * = interval excludes 0)."""
import json, sys
import numpy as np
from scipy import stats
sys.path.insert(0, "ours")
import run_contributions as rc
from analyse_contrib import load

ds = "CICIDS2017"
tuned = {k: tuple(v) for k, v in json.load(open(f"{rc.OUT}/tuned.json")).items()}
V = [("B0", "run", {}), ("C1", "run", dict(signal="error", tau=tuned["error_d0"])),
     ("C1_uniform_high", "run", dict(signal="error", level_mode="uniform_high")),
     ("C1b_boost", "run", dict(signal="error", tau=tuned["error_d0"], act_boost=True)),
     ("C1b_uniform_high", "run", dict(signal="error", level_mode="uniform_high", act_boost=True)),
     ("C1b_boost_only", "run", dict(signal="error", level_mode="uniform_high", act_boost=True, act_weights=False, act_replay=False, act_replace=False)),
     ("C1b_boost_fasttrust", "run", dict(signal="error", level_mode="uniform_high", act_boost=True, act_replay=False, act_replace=False))]
NEW = [6, 7, 8, 9, 10]
ALL = [1, 2, 3, 4, 5] + NEW

if __name__ == "__main__":
    if sys.argv[1] == "run":
        rc.pool([(ds, n, s, k, kw, None, "eval") for s in ALL for n, k, kw in V], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        seeds = [1, 2, 3, 4, 5] + NEW
        R = {n: load(ds, n, seeds)[0] for n, _, _ in V}
        print("paired difference to the baseline over 10 seeds (mean ±95% CI; * = excludes 0). Positive = higher.")
        for n in ("C1", "C1_uniform_high", "C1b_boost", "C1b_uniform_high", "C1b_boost_only", "C1b_boost_fasttrust"):
            print(f"\n{n}")
            for k, sc in [("acc", 100), ("macro_f1", 100), ("mcc", 1), ("worst_acc_w500", 100), ("drop", 100), ("recovery", 1)]:
                d = np.array([a[k] - b[k] for a, b in zip(R[n], R["B0"])]) * sc
                h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                print(f"  {k:15s} {d.mean():+.3f} ±{h:.3f}{' *' if abs(d.mean()) > h else ''}   (seeds better: {(d > 0).sum() if k not in ('drop', 'recovery') else (d < 0).sum()}/10)")
        for n in V:
            print(n[0], "recovery mean %.0f ± %.0f" % (np.mean([r['recovery'] for r in R[n[0]]]), np.std([r['recovery'] for r in R[n[0]]], ddof=1)))
