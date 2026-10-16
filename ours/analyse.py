"""Offline analysis of cached runs: metrics, window-weighting schemes, detectors, figures.
Run from sda/ after run_ext.py:  python experiments/analyse.py
"""
import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
                             cohen_kappa_score, roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score)
from river.drift import ADWIN, DDM, EDDM, KSWIN, PageHinkley, HDDM_A, HDDM_W

sys.path.insert(0, "ours")
from run_ext import DATA, SEEDS, BASE, load

OUT = "results/ext"
EPS = 0.001


def rolling_mean(a, W):
    cs = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(idx - W, 0)
    return (cs[idx] - cs[lo]) / (idx - lo)


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def full_metrics(y, pred, p1, lat, dr, last_label, size=None):
    n = len(y)
    minor = 0 if (y == 0).mean() < 0.5 else 1
    pm = p1 if minor == 1 else 1 - p1
    ym = (y == minor).astype(int)
    tn = ((pred == 0) & (y == 0)).sum(); tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    acc = accuracy_score(y, pred)
    prev = np.concatenate([[last_label], y[:-1]])                      # no-change classifier
    acc_nc = (prev == y).mean()
    maj = np.cumsum(y) / np.arange(1, n + 1) >= 0.5                    # running majority classifier
    acc_maj = (np.concatenate([[1], maj[:-1]]).astype(int) == y).mean()
    correct = (pred == y).astype(float)
    a500 = rolling_mean(correct, 500)[499:]
    a100 = rolling_mean(correct, 100)
    drops, recs = [], []
    for d in dr:
        pre = correct[max(0, d - 300):d].mean()
        seg = a100[d:d + 1000]
        tmin = int(np.argmin(seg))
        drop = pre - seg[tmin]
        drops.append(max(drop, 0))
        if drop < 0.01:
            recs.append(0)
        else:
            ok = np.where(seg[tmin:] >= pre - 0.01)[0]
            recs.append(tmin + ok[0] if len(ok) else 1000)
    return dict(
        acc=acc, prec=precision_score(y, pred, zero_division=0), rec=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0), macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        f1_minor=f1_score(ym, (pred == minor).astype(int), zero_division=0),
        bal_acc=balanced_accuracy_score(y, pred), gmean=float(np.sqrt(tpr * tnr)), mcc=matthews_corrcoef(y, pred),
        kappa=cohen_kappa_score(y, pred), kappa_t=(acc - acc_nc) / (1 - acc_nc + 1e-12),
        kappa_m=(acc - acc_maj) / (1 - acc_maj + 1e-12),
        fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1),
        auc=roc_auc_score(y, p1), ap_minor=average_precision_score(ym, pm),
        brier=brier_score_loss(y, p1), ece=ece(y, p1),
        worst_acc_w500=a500.min(), p5_acc_w500=np.percentile(a500, 5),
        drop=float(np.mean(drops)), recovery=float(np.mean(recs)),
        lat_ms=lat.mean() * 1e3, lat_p99_ms=np.percentile(lat, 99) * 1e3, thr=len(lat) / lat.sum(),
        model_kb=(size or -1) / 1024)


def load_npz(kind, ds, name, s):
    return np.load(f"{OUT}/{kind}__{ds}__{name.replace('/', '_')}__s{s}.npz")


# ---------------------------------------------------------------- ensemble weighting schemes (offline)
def win_err(E, W, inclusive):
    """E: (k,n) error indicators. window mean error of each learner before (or incl.) t; W=None -> whole history."""
    k, n = E.shape
    cs = np.concatenate([np.zeros((k, 1)), np.cumsum(E, 1)], 1)
    t = np.arange(n)
    hi = t + (1 if inclusive else 0)
    lo = np.zeros(n, int) if W is None else np.maximum(hi - W, 0)
    cnt = hi - lo
    out = (cs[:, hi] - cs[:, lo]) / np.maximum(cnt, 1)
    out[:, cnt == 0] = 0.5
    return out


def fading_err(E, a):
    k, n = E.shape
    out = np.full((k, n), 0.5)
    m = np.full(k, 0.5)
    for t in range(n):
        out[:, t] = m
        m = a * m + (1 - a) * E[:, t]
    return out


def combine(P, e, mode="inv"):
    if mode == "inv":
        w = 1 / (e + EPS)
    elif mode == "leader":
        w = (e == e.min(0, keepdims=True)).astype(float)
    elif mode == "softmax":
        w = np.exp(-(e - e.min(0, keepdims=True)) / 0.02)
    return (w * P).sum(0) / w.sum(0)


def ensemble_variants(P, E, y):
    n = P.shape[1]
    V = {}
    V["equal-mean"] = P.mean(0)
    votes = (P >= 0.5).sum(0)
    V["majority-vote"] = np.where(votes == 2, P.mean(0), (votes > 2).astype(float))
    V["PWPAE history (paper, leaks y_t)"] = combine(P, win_err(E, None, True))
    V["PWPAE history (causal)"] = combine(P, win_err(E, None, False))
    # MSANA: last 10% of processed samples, equal weights for first 1000
    e = np.zeros_like(E, float)
    cs = np.concatenate([np.zeros((E.shape[0], 1)), np.cumsum(E, 1)], 1)
    for t in range(1000, n):
        lo = round(0.9 * t)
        e[:, t] = (cs[:, t] - cs[:, lo]) / max(t - lo, 1)
    V["W-PWPAE last-10% (MSANA)"] = combine(P, e)
    for W in (100, 500, 1000, 2000):
        V[f"sliding W={W}"] = combine(P, win_err(E, W, False))
    for a in (0.99, 0.999, 0.9999):
        V[f"fading a={a}"] = combine(P, fading_err(E, a))
    V["leader W=500"] = combine(P, win_err(E, 500, False), "leader")
    V["softmax W=500"] = combine(P, win_err(E, 500, False), "softmax")
    # reset window at EDDM alarms of the equal-weight ensemble error
    eq_err = ((P.mean(0) >= 0.5).astype(int) != y).astype(float)
    det = EDDM(); alarms = []
    for t, v in enumerate(eq_err):
        if det.update(float(v))[0]:
            alarms.append(t)
    start = np.zeros(n, int); last = 0; ai = 0
    for t in range(n):
        while ai < len(alarms) and alarms[ai] < t:
            last = alarms[ai]; ai += 1
        start[t] = last
    e = np.zeros_like(E, float)
    for t in range(n):
        lo = start[t] if t - start[t] >= 50 else max(0, t - 500)
        e[:, t] = (cs[:, t] - cs[:, lo]) / max(t - lo, 1) if t > lo else 0.5
    V["since-last-EDDM-alarm"] = combine(P, e)
    return V


# ---------------------------------------------------------------- detectors on the error stream
def detector_study(err, dr, tol):
    dets = {"ADWIN": ADWIN, "DDM": DDM, "EDDM": EDDM, "HDDM_A": HDDM_A, "HDDM_W": HDDM_W,
            "KSWIN": KSWIN, "PageHinkley": PageHinkley}
    rows = []
    for name, cls in dets.items():
        d = cls(); al = []
        for t, v in enumerate(err):
            r = d.update(float(v))
            if (r[0] if isinstance(r, tuple) else getattr(d, "change_detected", False)):
                al.append(t)
        al = np.array(al)
        delays, hit = [], 0
        for g in dr:
            m = al[(al >= g) & (al < g + tol)]
            if len(m):
                hit += 1; delays.append(m[0] - g)
        inwin = np.zeros(len(al), bool)
        for g in dr:
            inwin |= (al >= g) & (al < g + tol)
        rows.append(dict(detector=name, alarms=len(al), detected=f"{hit}/{len(dr)}",
                         mean_delay=float(np.mean(delays)) if delays else np.nan,
                         false_alarms=int((~inwin).sum()), fa_per_1k=1000 * (~inwin).sum() / len(err)))
    return pd.DataFrame(rows)


def agg(df, by):
    num = df.select_dtypes("number").columns.difference(by + ["seed"])
    g = df.groupby(by, sort=False)[list(num)]
    m, s = g.mean(), g.std()
    return m, s


def md(m, s, cols, fmt=3, pct=()):
    h = "| method | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for i in m.index:
        cells = []
        for c in cols:
            v, sd = m.loc[i, c], s.loc[i, c]
            if c in pct:
                v, sd = v * 100, sd * 100
            cells.append(f"{v:.{fmt}f}±{sd:.{fmt}f}" if not np.isnan(sd) else f"{v:.{fmt}f}")
        h += f"| {i if isinstance(i, str) else i[-1]} | " + " | ".join(cells) + " |\n"
    return h


if __name__ == "__main__":
    summary = []
    allrows = {"sample": [], "ens": [], "base": []}
    det_rows = []
    for ds in DATA:
        tr, ytr, te, yte, dr = load(ds)
        last = int(ytr[-1])
        # --- sampling
        for k in ["none", "RUS-40/60", "ROS-40/60", "RUS-50/50", "ROS-50/50", "oob", "uob", "sw", "smote"]:
            for s in SEEDS:
                z = load_npz("sample", ds, k, s)
                allrows["sample"].append(dict(ds=ds, method=k, seed=s, **full_metrics(z["y"], z["pred"], z["p1"], z["lat"], dr, last, int(z["size"]))))
        # --- base learners + ensembles + detectors
        for s in SEEDS:
            Z = [load_npz("base", ds, b, s) for b in BASE]
            P = np.stack([z["p1"] for z in Z]); pr = np.stack([z["pred"] for z in Z])
            E = (pr != yte[None]).astype(float)
            for b, z in zip(BASE, Z):
                allrows["base"].append(dict(ds=ds, method=b, seed=s, **full_metrics(z["y"], z["pred"], z["p1"], z["lat"], dr, last, int(z["size"]))))
            lat_sum = sum(z["lat"] for z in Z)
            for name, p in ensemble_variants(P, E, yte).items():
                allrows["ens"].append(dict(ds=ds, method=name, seed=s, **full_metrics(yte, (p >= 0.5).astype(int), p, lat_sum, dr, last)))
            tol = 300 if ds == "IoTID20" else 1000
            dd = detector_study(E[0], dr, tol); dd["ds"] = ds; dd["seed"] = s
            det_rows.append(dd)
        # --- time-series figure
        fig, ax = plt.subplots(figsize=(11, 4))
        for k, c in [("none", "k"), ("ROS-40/60", "tab:blue"), ("oob", "tab:green"), ("smote", "tab:red")]:
            ra = np.mean([rolling_mean((load_npz("sample", ds, k, s)["pred"] == yte).astype(float), 200) for s in SEEDS], 0)
            ax.plot(ra, c, lw=1, label=k)
        for g in dr:
            ax.axvline(g, color="r", ls="--", lw=0.8)
        ax.set_ylim(0.85, 1.005); ax.set_xlabel("test-stream index"); ax.set_ylabel("rolling accuracy (W=200)")
        ax.set_title(f"{ds}: ARF-ADWIN under sampling strategies (mean of 3 seeds; red = annotated drifts)"); ax.legend()
        fig.tight_layout(); fig.savefig(f"results/fig_timeseries_{ds}.png", dpi=130); plt.close(fig)

    out = ["# Extension experiment results (auto-generated by analyse.py)\n",
           f"Seeds: {SEEDS}. Values are mean±std over seeds (single-run std is 0). Percent metrics shown ×100.\n"]
    pct = ("acc", "prec", "rec", "f1", "macro_f1", "f1_minor", "bal_acc", "gmean", "fpr", "fnr", "auc", "ap_minor")
    cols1 = ["acc", "f1", "macro_f1", "f1_minor", "bal_acc", "gmean", "mcc", "kappa", "kappa_t", "kappa_m", "auc", "ap_minor", "fpr", "fnr", "brier", "ece"]
    cols2 = ["worst_acc_w500", "p5_acc_w500", "drop", "recovery", "lat_ms", "lat_p99_ms", "thr", "model_kb"]
    for key, title in [("base", "Base learners (PWPAE members)"), ("ens", "Ensemble weighting / window schemes (same 4 cached learners)"),
                       ("sample", "Sampling strategies on ARF-ADWIN")]:
        df = pd.DataFrame(allrows[key])
        df.to_csv(f"results/{key}_all.csv", index=False)
        for ds in DATA:
            d = df[df.ds == ds]
            m, s = agg(d.drop(columns="ds"), ["method"])
            order = list(dict.fromkeys(d.method))
            m, s = m.loc[order], s.loc[order]
            out.append(f"\n## {title} — {ds}\n\n**Predictive / imbalance / calibration**\n\n" + md(m, s, cols1, 3, pct))
            out.append("\n**Drift-response and cost** (drop = mean accuracy dip after annotated drift; recovery = samples to regain pre-drift accuracy-1pt, capped 1000; thr = samples/s)\n\n" + md(m, s, cols2, 3, ("worst_acc_w500", "p5_acc_w500", "drop")))
    dd = pd.concat(det_rows)
    dd.to_csv("results/detectors_all.csv", index=False)
    for ds in DATA:
        t = dd[dd.ds == ds].groupby("detector", sort=False).agg(alarms=("alarms", "mean"), mean_delay=("mean_delay", "mean"),
                                                                 false_alarms=("false_alarms", "mean"), fa_per_1k=("fa_per_1k", "mean"),
                                                                 detected=("detected", lambda x: ",".join(x)))
        out.append(f"\n## Drift detectors on the ARF-ADWIN error stream — {ds}\n\n" + t.round(2).to_markdown() + "\n")
    # bar figures
    for key, fn in [("sample", "fig_sampling.png"), ("ens", "fig_windows.png")]:
        df = pd.DataFrame(allrows[key])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5 if key == "sample" else 6.5))
        for ax, ds in zip(axes, DATA):
            d = df[df.ds == ds]
            g = d.groupby("method", sort=False)[["macro_f1", "mcc"]].agg(["mean", "std"])
            x = np.arange(len(g))
            ax.bar(x - 0.2, g[("macro_f1", "mean")], 0.4, yerr=g[("macro_f1", "std")], label="macro-F1")
            ax.bar(x + 0.2, g[("mcc", "mean")], 0.4, yerr=g[("mcc", "std")], label="MCC")
            ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=70, ha="right", fontsize=8)
            lo = min(g[("mcc", "mean")].min(), g[("macro_f1", "mean")].min())
            ax.set_ylim(max(lo - 0.03, 0), 1.0); ax.set_title(ds); ax.legend()
        fig.tight_layout(); fig.savefig(f"results/{fn}", dpi=130); plt.close(fig)
    open("results/ext_summary.md", "w", encoding="utf8").write("\n".join(out))
    print("done")
