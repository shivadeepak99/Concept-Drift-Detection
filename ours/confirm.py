"""Confirmation on FRESH seeds (never used for tuning or for choosing the variant) of the best variant found on seeds 1-10,
plus checks that it does no harm on IoTID20 and what it does when labels are 200 flows late.
   python ours/confirm.py run [workers]  |  python ours/confirm.py report
"""
import sys
import numpy as np
from scipy import stats
sys.path.insert(0, "ours")
import run_contributions as rc
from analyse_contrib import load

BEST = dict(signal="error", level_mode="uniform_high", act_boost=True)
FRESH = list(range(21, 31))


def specs():
    S = []
    for s in FRESH:
        S += [("CICIDS2017", "B0", s, "run", {}, None, "eval"), ("CICIDS2017", "C1b_uniform_high", s, "run", BEST, None, "eval")]
    for s in (1, 2, 3):
        S += [("IoTID20", "C1b_uniform_high", s, "run", BEST, None, "eval")]
    for s in (1, 2, 3, 4, 5):
        S += [("CICIDS2017", "C1b_uniform_high_d200", s, "run", {**BEST, "delay": 200}, None, "eval")]
    return S


def paired(a, b, k, sc=1):
    d = np.array([x[k] - y[k] for x, y in zip(a, b)]) * sc
    h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), h, int((d > 0).sum()), len(d)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        rc.pool(specs(), int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        def rep(title, ds, a, b, seeds_a, seeds_b=None):
            A, B = load(ds, a, seeds_a)[0], load(ds, b, seeds_b or seeds_a)[0]
            print(f"\n{title}  (n={len(A)} paired seeds; mean difference ±95% CI, * = excludes 0)")
            for k, sc in [("acc", 100), ("macro_f1", 100), ("mcc", 1), ("worst_acc_w500", 100), ("drop", 100), ("recovery", 1)]:
                m, h, w, n = paired(A, B, k, sc)
                print(f"  {k:15s} {m:+.3f} ±{h:.3f}{' *' if abs(m) > h else ''}   better in {w if k not in ('drop','recovery') else n-w}/{n} seeds")
        rep("CICIDS2017, fresh seeds 21-30: always-react-hard + boost  vs  baseline", "CICIDS2017", "C1b_uniform_high", "B0", FRESH)
        rep("IoTID20, seeds 1-3: same variant vs baseline", "IoTID20", "C1b_uniform_high", "B0", [1, 2, 3])
        rep("CICIDS2017, labels 200 flows late, seeds 1-5: same variant vs baseline with the same delay", "CICIDS2017", "C1b_uniform_high_d200", "B0_d200", [1, 2, 3, 4, 5])
