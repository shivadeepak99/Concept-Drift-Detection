"""Our version of PWPAE: the baseline loop plus a drift-severity-aware response layer.

Everything not marked OURS is the baseline's logic (4 online learners, error-weighted probability averaging,
predict -> learn on every flow). Run from the repo root; needs River 0.8.0 (see README).

Variants (all share one engine, switched by `cfg`):
  baseline        signal=None                      -> plain PWPAE (weights from whole-history error, causal)
  severity        signal="error"                   -> alarm (EDDM on ensemble error) -> severity S -> low/medium/high reaction   [contribution 1]
  uncertainty     signal="uncertainty"             -> alarm (ADWIN on model uncertainty, needs no labels)                       [contribution 2]
  delay=D         labels arrive D flows late       -> tests how each variant copes with delayed labels                          [contribution 2]
  run_gated(...)  one cheap forest, full ensemble only after an alarm                                                          [contribution 3]
"""
import time
from collections import deque
import numpy as np
from river import ensemble
from river.drift import ADWIN, EDDM

import run_ext as rx   # make_base(), load()

KINDS = ["ARF-ADWIN", "SRP-ADWIN", "ARF-DDM", "SRP-DDM"]
EPS_W = 0.001   # same epsilon as the baseline weights: w = 1 / (error + 0.001)

CFG = dict(
    signal=None,             # None | "error" | "uncertainty"
    delay=0,                 # labels revealed this many flows after the prediction
    level_mode="severity",   # "severity" | "uniform_medium" | "uniform_high"
    tau=(1.0, 3.0),          # S < tau1 low, < tau2 medium, else high
    act_weights=True,        # medium/high: judge models on the last `win` flows for the next `hold` flows
    act_replay=True,         # medium/high: re-teach every model on the last `replay` labelled flows
    act_replace=True,        # high: replace the worst model by a fresh one trained on the last `rebuild` flows
    act_boost=False,         # OURS (C1b): medium/high: learn every NEW flow several times for a while (boost_k flows-repeats for boost_hold flows)
    boost_k=(2, 4), boost_hold=(500, 1000),
    eps_s=0.01, recent=100, base=300,   # severity window: last `recent` flows vs the `base` flows before them
    hold=1000, win=200, replay=200, buffer=1000, rebuild=500, cooldown=200,
)


def severity(sig, recent, base, eps):
    """OURS. S = |recent - normal| / (normal + eps). None until enough history exists."""
    if len(sig) < recent + base // 2:
        return None
    a = np.asarray(sig[-recent:], float)
    b = np.asarray(sig[-(recent + base):-recent], float)
    return abs(a.mean() - b.mean()) / (b.mean() + eps)


def level_of(S, mode, tau):
    if mode == "uniform_medium":
        return "medium"
    if mode == "uniform_high":
        return "high"
    return "low" if S < tau[0] else ("medium" if S < tau[1] else "high")


def run(ds, seed, n_stop=None, **over):
    c = {**CFG, **over}
    tr, ytr, te, yte, _ = rx.load(ds)
    if n_stop:
        te, yte = te[:n_stop], yte[:n_stop]
    models = [rx.make_base(k, seed) for k in KINDS]
    for x, y in zip(tr, ytr):
        for m in models:
            m.learn_one(x, int(y))
    n, K, D = len(te), len(KINDS), c["delay"]
    E = np.zeros((K, n)); EE = np.zeros(n)          # per-model / ensemble error indicators, filled when a label is revealed
    cum = np.zeros(K); nrev = 0
    p1 = np.zeros(n); pred = np.zeros(n, int); lat = np.zeros(n)
    pend, buf, u_hist = deque(), deque(maxlen=c["buffer"]), []
    fast_until, last_react, n_replaced = -1, -10 ** 9, 0
    boost_until, boost_n = -1, 1
    eddm, adw = EDDM(), ADWIN(delta=0.002)
    events = []
    pc = time.perf_counter
    for i, (x, y) in enumerate(zip(te, yte)):
        t0 = pc()
        probs = np.array([m.predict_proba_one(x).get(1, 0.0) for m in models])
        if nrev == 0:
            w = np.ones(K)
        else:
            e = E[:, max(0, nrev - c["win"]):nrev].mean(1) if i < fast_until else cum / nrev   # OURS: fast trust after a reaction
            w = 1 / (e + EPS_W)
        p = float((w * probs).sum() / w.sum()); pr = int(p >= 0.5)
        p1[i], pred[i] = p, pr
        pend.append((x, int(y), probs, pr))
        u = min(p, 1 - p); u_hist.append(u)
        alarm = False
        if c["signal"] == "uncertainty":
            adw.update(u); alarm = bool(adw.change_detected)
        if len(pend) > D:                              # the label of flow i-D arrives now
            x0, y0, pv, pr0 = pend.popleft()
            for _ in range(boost_n if i < boost_until else 1):          # OURS (C1b): repeat learning of new flows after a reaction
                for m in models:
                    m.learn_one(x0, y0)
            buf.append((x0, y0))
            ej = ((pv >= 0.5).astype(int) != y0).astype(float)
            E[:, nrev] = ej; cum += ej; EE[nrev] = float(pr0 != y0); nrev += 1
            if c["signal"] == "error":
                alarm = bool(eddm.update(float(pr0 != y0))[0])
        if alarm and i - last_react >= c["cooldown"]:     # ---------------- OURS: severity-scaled reaction
            sig = EE[:nrev] if c["signal"] == "error" else u_hist
            S = severity(sig, c["recent"], c["base"], c["eps_s"])
            if S is not None:
                lvl = level_of(S, c["level_mode"], c["tau"])
                last_react = i
                events.append((i, round(float(S), 3), lvl))
                if lvl != "low":
                    if c["act_weights"]:
                        fast_until = i + c["hold"]
                    if c["act_boost"]:
                        j = 1 if lvl == "high" else 0
                        boost_until, boost_n = i + c["boost_hold"][j], c["boost_k"][j]
                    if c["act_replay"]:
                        for xx, yy in list(buf)[-c["replay"]:]:
                            for m in models:
                                m.learn_one(xx, yy)
                    if lvl == "high" and c["act_replace"] and nrev > c["win"]:
                        worst = int(np.argmax(E[:, nrev - c["win"]:nrev].mean(1)))
                        new = rx.make_base(KINDS[worst], seed + 1000 + n_replaced)
                        for xx, yy in list(buf)[-c["rebuild"]:]:
                            new.learn_one(xx, yy)
                        models[worst] = new; n_replaced += 1
                        others = [j for j in range(K) if j != worst]      # neutral history for the newcomer
                        E[worst, :nrev] = E[others, :nrev].mean(0); cum[worst] = cum[others].mean()
        lat[i] = pc() - t0
    return dict(p1=p1, pred=pred, lat=lat, y=yte, events=events, replaced=n_replaced)


def run_single(ds, seed, trees=10, n_stop=None):
    """Reference for contribution 3: one ARF-ADWIN forest, no ensemble."""
    tr, ytr, te, yte, _ = rx.load(ds)
    if n_stop:
        te, yte = te[:n_stop], yte[:n_stop]
    from river.drift import ADWIN as A
    m = ensemble.AdaptiveRandomForestClassifier(n_models=trees, drift_detector=A(), seed=seed)
    p1, pred, lat = rx.prequential(m, tr, ytr, te, yte)
    return dict(p1=p1, pred=pred, lat=lat, y=yte, events=[], replaced=0)


def run_gated(ds, seed, trees=10, hold=600, buffer=1000, n_stop=None, win=200):
    """Contribution 3: the cheap forest works alone. An error alarm wakes three more models (trained on the last
    `buffer` labelled flows), the four vote for `hold` flows, then the extra three are dropped again."""
    tr, ytr, te, yte, _ = rx.load(ds)
    if n_stop:
        te, yte = te[:n_stop], yte[:n_stop]
    prim = ensemble.AdaptiveRandomForestClassifier(n_models=trees, drift_detector=ADWIN(), seed=seed)
    for x, y in zip(tr, ytr):
        prim.learn_one(x, int(y))
    n = len(te); p1 = np.zeros(n); pred = np.zeros(n, int); lat = np.zeros(n)
    extra, extra_E_rows, active_until, ne = [], [], -1, 0
    buf = deque(maxlen=buffer); eddm = EDDM(); events = []; active = np.zeros(n, bool)
    pc = time.perf_counter
    for i, (x, y) in enumerate(zip(te, yte)):
        t0 = pc()
        pp = prim.predict_proba_one(x).get(1, 0.0)
        if extra:
            probs = np.array([pp] + [m.predict_proba_one(x).get(1, 0.0) for m in extra])
            if ne >= 50:
                e = np.array([np.mean(r[-win:]) for r in extra_E_rows])
                w = 1 / (e + EPS_W)
            else:
                w = np.ones(len(probs))
            p = float((w * probs).sum() / w.sum()); active[i] = True
        else:
            probs, p = None, pp
        pr = int(p >= 0.5); p1[i], pred[i] = p, pr
        prim.learn_one(x, int(y)); buf.append((x, int(y)))
        if extra:
            for m in extra:
                m.learn_one(x, int(y))
            for r, v in zip(extra_E_rows, probs):
                r.append(float(int(v >= 0.5) != int(y)))
            ne += 1
            if i >= active_until:
                extra, extra_E_rows, ne = [], [], 0
        alarm = bool(eddm.update(float(pr != int(y)))[0])
        if alarm:
            if not extra:
                extra = [rx.make_base(k, seed + 2000 + i) for k in KINDS[1:]]
                for xx, yy in list(buf):
                    for m in extra:
                        m.learn_one(xx, yy)
                extra_E_rows = [[] for _ in range(len(extra) + 1)]; ne = 0
                events.append((i, 0.0, "wake"))
            active_until = i + hold
        lat[i] = pc() - t0
    return dict(p1=p1, pred=pred, lat=lat, y=yte, events=events, replaced=0, active=active)
