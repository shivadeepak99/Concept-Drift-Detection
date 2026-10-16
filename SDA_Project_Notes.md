# SDA Mini-Project — Status, Reproduction and Research Notes

**Project:** Adaptive Concept Drift Detection and Response for Real-Time IIoT Network Data Streams (IOE413 Streaming Data Analytics)
**Team:** Sidhaarth Mohandas (lead), Abin Tomy, Shanigram Shiva Deepak, Uday Kiran
**Prepared:** 15 Oct 2026, from the project PDFs and a fresh run of the authors' code. All numbers below come from files in this folder (`results/`, `experiments/`) unless a source is cited.

---

## 1. Where the project stands

**What the project is (plain words).** Network-traffic streams from IoT/IIoT devices change over time (new attacks, new devices). A model trained on old traffic degrades — "concept drift". The team will **reproduce PWPAE** (an ensemble of four online learners whose votes are weighted by recent accuracy) as the baseline and add a **drift-severity-aware response**: measure how big the change is (S = |E_recent − E_baseline| / (E_baseline + ε)) and react gently to small drifts, aggressively (re-weight, replace weak learners) to large ones. MSANA (the paper you were given as `SDA_Industrial_Iot.pdf`) and CDDIA are related work only. Planned stack: Python replay → Kafka → Flink → River → InfluxDB → Grafana. Datasets: IoTID20 (primary), CICIDS2017 (validation).

**Status as evidenced by this folder: proposal stage only.** The folder holds just `Project_Proposal.pdf` and `SDA_Industrial_Iot.pdf` (the MSANA paper, arXiv 2210.01985). There is no code, result, log or report from the team. I cannot see chat history, so "not evidenced" below means *I found no proof*, not *it was not done* — confirm with Sidhaarth.

| Proposal plan item (Sec. 11) | Evidence | Status |
|---|---|---|
| Reproduce PWPAE baseline, record versions/deviations | Done in this session (Sec. 2) | **Done (by this work)** |
| Kafka replay of IoTID20; Kafka→Flink→online-model flow | none | Not evidenced |
| Implement drift-severity-aware layer | none (pilot severity numbers in Sec. 3.4) | Not started / not evidenced |
| Baseline vs enhancement on same stream | none | Not started |
| Repeat on CICIDS2017 | baseline run done (Sec. 2) | Baseline only |
| Compare predictive / drift-response / system cost | metric code written (Sec. 3.1) | Tooling ready, no comparison yet |
| Grafana visualisation | none | Not evidenced |

**Three things the new results mean for the proposal** (details in Secs. 2–4):
1. **IoTID20 shows almost no detectable drift.** Seven standard detectors run on the learner's error stream fire 0–2 times on the IoTID20 sample, and never reliably at the "drift points" the authors mark (270, 600). All six marked CICIDS2017 drifts are real error bursts. Hypotheses H1/H3 (severity-aware recovery) can only be tested meaningfully on CICIDS2017 unless drift is injected or the full IoTID20 is used.
2. **Differences between published methods are smaller than run-to-run noise.** Same config, same notebook, run twice: 97.96 % vs 98.58 % accuracy on IoTID20 (ARF-ADWIN, MSANA notebook cells 15 and 24, unseeded). Across 3 seeds the std of ensemble accuracy is 0.2 pp on IoTID20; PWPAE vs MSANA differ by 0.02–0.05 pp in the paper. Any claimed improvement needs multiple seeds and an interval.
3. **The streaming stack is the largest unbuilt part.** PWPAE processes ≈110–176 samples/s on CPU (paper: 110–134; mine: ≈140–176 for the four learners together), so Kafka/Flink are not the bottleneck, but wiring River (Python) into Flink (PyFlink) is a real integration risk worth a spike early.

---

## 2. Running the authors' implementations

**Setup.** The repos pin `river 0.1.0` (a Python 3.6 build shipped as a zip; not installable on Windows/py3.12). Modern River (0.26) no longer has `KNNADWINClassifier`/`SAMKNNClassifier` and moved ARF out of `ensemble`, so I used **River 0.8.0 on Python 3.8** (still has the notebooks' APIs, including EDDM returning `(drift, warning)`). All four notebooks were executed with `nbconvert`; outputs are in `results/executed/`.

**Reproduction vs published numbers** (accuracy / precision / recall / F1, %; published values = Table I of the MSANA paper):

| Model | Dataset | Published | My notebook run | My 3-seed mean (script) |
|---|---|---|---|---|
| ARF-ADWIN | IoTID20 | 98.38 / 98.55 / 99.75 / 99.15 | 98.45 / 98.60 / 99.77 / 99.18 | acc 98.39±0.23, F1 99.15±0.12 |
| ARF-ADWIN | CICIDS2017 | 98.95 / 97.50 / 96.12 / 96.81 | 98.81 / 97.42 / 95.64 / 96.52 | acc 98.76±0.05, F1 96.39±0.14 |
| PWPAE | IoTID20 | 98.90 / 98.86 / 99.98 / 99.42 | 99.16 / 99.23 / 99.89 / 99.56 | acc 98.97±0.21, F1 99.46±0.11 |
| PWPAE | CICIDS2017 | 99.27 / 98.40 / 97.19 / 97.79 | 99.14 / 98.42 / 96.57 / 97.49 | acc 99.15±0.05, F1 97.50±0.14 |
| MSANA (W-PWPAE) | IoTID20 | 98.88 / 98.88 / 99.94 / 99.41 | 97.58 / 97.68 / 99.81 / 98.73 (default cell); 97.87 / 98.09 / 99.68 / 98.88 (with over-sampling) | — |
| MSANA (W-PWPAE) | CICIDS2017 | 99.32 / 98.31 / 97.59 / 97.95 | 99.11 / 98.02 / 96.52 / 97.26 (notebook default; PWPAE in the same notebook: 98.81 / 97.51 / 95.19 / 96.33) — direction reproduces (MSANA > PWPAE by ≈0.3 pp) | — |

PWPAE and its base learners reproduce within ≈0.1–0.3 pp. Note the MSANA repo's CICIDS2017 file has 14,155 rows (`cic_0.005km_78.csv`) while the paper text says 28,303 records, so that row is not the same stream as the published one. **MSANA on IoTID20 does not reproduce** from the released notebook (≈1 pp lower and *below* PWPAE in the same notebook), most likely because the notebook's default ensemble (ARF-ADWIN, KNN-ADWIN, ARF-EDDM, OPA) is not the one the paper reports for IoTID20 (EFDT + KNN-ADWIN followers), and EFDT crashes under River 0.8.0 (below). Latency: my four learners sum to ≈5.7–7.1 ms/sample vs 7.4–9.1 ms published (different CPU; includes the learning step).

**Reproducibility problems found (useful for the report):**
- `river 0.1.0` bundled only as a py3.6 zip; no `requirements.txt`; current River breaks the notebooks.
- `ExtremelyFastDecisionTreeClassifier` raises `AttributeError: 'NoneType' object has no attribute 'merit'` under River 0.8.0, so the EFDT cell aborts and the notebooks' final comparison plot cell fails with `NameError` (IoTID20 notebooks; also MSANA-CICIDS2017). Published EFDT numbers therefore cannot be reproduced here.
- Models are unseeded: results differ run to run (see Sec. 1, point 2).
- The PWPAE notebook computes ensemble weights *after* updating the metric with the current true label (the weights for sample *t* see whether each learner was right on sample *t*). I measured the effect: +0.006 pp on IoTID20, +0.004 pp on CICIDS2017 — negligible, but it is not strictly test-then-train. The MSANA notebook slices `[:i]` and does not leak.
- The sample files are *k-means cluster-sampled* subsets (PWPAE paper; file names suggest 1 % / 0.5 %) — 6,252 and 28,303 rows, not the original ≈625 k / ≈2.8 M flows. How that sampling bends the drift structure is not stated.
- "Drift points" in the plots (`dr = [0,270,600]`, `[0,9900,12600,14200,15800,18800,21600]`) are hand-marked, not ground truth. On CICIDS2017 there are error bursts near samples ≈1,800 and ≈3,600 that are not marked, while 9,900 shows no accuracy change at all (`results/fig_timeseries_CICIDS2017.png`).

**Extension runs.** `experiments/run_ext.py` re-runs the four PWPAE learners (3 seeds × 2 datasets) and ARF-ADWIN under 9 sampling strategies, caching predictions; `experiments/analyse.py` computes metrics, window schemes and detector results offline. Full tables: `results/ext_summary.md`; raw rows: `results/*_all.csv`.

---

## 3. Lead's question 1 — new metrics and making results interpretable

### 3.1 Metrics worth adding, with evidence from the runs

The papers report accuracy, precision, recall, F1 (positive class), mean latency and throughput. On these streams that hides a lot:

| Metric | Why add it | What it shows here | Source |
|---|---|---|---|
| **Macro-F1 / minority-class F1** | On IoTID20 label 1 is 94 % of data, so the paper's F1 is the majority class's F1. | ARF-ADWIN IoTID20: F1(label 1) 99.2 % but minority-class F1 **84.3 %**, macro-F1 91.7 %. PWPAE: 99.5 % vs 90.2 % / 94.8 %. | Chicco & Jurman 2020 |
| **MCC** (+ balanced accuracy, G-mean) | Uses all four confusion-matrix cells; hard to inflate on imbalance. | IoTID20 MCC 0.84 (ARF-ADWIN) → 0.90 (PWPAE); CICIDS2017 0.956 → 0.970. | Chicco & Jurman 2020 |
| **FPR / FNR separately** | Security cost is asymmetric; accuracy hides it. | IoTID20: ARF-ADWIN lets **24 %** of normal-class (minority) flows through as the other class; PWPAE 17 %. CICIDS2017 FNR 4.0 % → 3.2 %. | — |
| **Kappa-temporal (κ⁺) and Kappa-M** | Compare against a "repeat the last label" / majority classifier, because labels in network streams come in runs. | CICIDS2017 ARF-ADWIN: κ = 0.956 but **κ⁺ = 0.657** — a no-change predictor already scores well, so 98.8 % accuracy is partly temporal persistence. | Bifet et al. 2013; Žliobaitė et al. 2015; Brzezinski & Stefanowski 2017 |
| **Windowed / worst-case accuracy** (W = 500) | Overall accuracy averages away the post-drift dips the paper is about. | CICIDS2017: overall 98.8 %, **worst 500-sample window 88.9 %**; PWPAE 99.1 % vs 93.7 %. | Gama et al. 2013 |
| **Recovery time & accuracy drop per drift** | The proposal's own H1 metric; not in any base paper. | ≈300–350 samples to regain pre-drift accuracy −1 pp (ARF-ADWIN, both datasets); mean dip 5 pp (IoT) / 11 pp (CIC). | proposal Sec. 10; Gama et al. 2013 |
| **Detector metrics: MTD, MDR, MTFA / false-alarm rate** | Separate "did we see the drift" from "did accuracy hold". | CICIDS2017 (ARF-ADWIN error stream): EDDM 7.7 alarms, 3.3 false, mean delay 164; ADWIN 14 alarms, 6 false, delay 298; HDDM/KSWIN/Page-Hinkley ≤1 alarm. IoTID20: ≈0 alarms. | arXiv 2605.31186; arXiv 2606.07789 |
| **Prequential AUC / PR-AUC** | Threshold-free; robust to class-ratio drift. (I computed whole-stream ROC-AUC and PR-AUC, not the windowed prequential version.) | CICIDS2017 AUC 99.7–99.8 % — far less separation between methods than accuracy suggests. | Brzezinski & Stefanowski 2017 |
| **Calibration: Brier score, ECE** | PWPAE averages probabilities, so probability quality matters and feeds alerts/thresholds. | Equal-mean ECE ≈ 0.02; resampling worsens ECE and Brier, up to ~10× (Sec. 5). | — |
| **Tail latency (p95/p99) and model size / RAM-hours** | Papers report only mean latency; drift handling causes spikes (paper: up to 16–38 ms). | PWPAE ensemble p99 11–14 ms vs mean 6–7 ms; single ARF p99 ≈ 2 ms. Pickle size 0.1–0.5 MB per learner (a proxy — true RAM needs process-level monitoring). | Bifet et al. 2010 (RAM-hours); Haug et al. 2022 |
| **Multi-seed mean ± std, paired test** | Differences between methods are within noise (Sec. 1). | IoTID20 ensemble accuracy std 0.2 pp across 3 seeds. | — |

### 3.2 Making the models more understandable / interpretable (what could be added)

1. **Drift timeline with explanations per event:** for every detector alarm show error before/after, the severity score S, the detector that fired and its delay. Cheap, and it is the visual story for the proposal's severity idea (Grafana panel).
2. **Ensemble weight trajectories:** plot w_j(t) for ARF-ADWIN, ARF-DDM, SRP-ADWIN, SRP-DDM. PWPAE's weights are directly interpretable ("which learner does the ensemble trust now"); in the runs they range 0.02–0.83 over the stream and average 0.23–0.28 per learner.
3. **Feature-level drift explanation:** which features changed at each drift — model-based explanations (train a classifier to separate before/after, then explain it with SHAP or similar; Hinder et al. 2023) and the CDDIA idea of SHAP on drift samples. Report top-k features per drift segment.
4. **Per-segment analysis between drifts:** confusion matrix, per-attack-type recall per segment. The sample files are binary, but the raw datasets have attack types; "which attack family arrived at drift 3" is the interpretable output users want.
5. **Calibrated probabilities / uncertainty:** show the ensemble's agreement (spread across the four learners) and calibration curve — low agreement often precedes a drift.
6. **Unlabelled operation:** the base papers assume the true label arrives instantly after each prediction. Report results with a label delay or feedback-rate budget (surveys on delayed/partially labelled streams, arXiv 2106.09170).

---

## 4. Lead's question 2 — what other windows could be added

**Caveat on scope.** MSANA's window does two jobs: (a) *weights* the ensemble by recent error, (b) *retrains* learners on the recent window after a drift. My experiments isolate job (a) only: the four base learners are the same cached predictions, and only the weighting window changes. Job (b) is not tested.

### 4.1 Windows tried (weighting schemes over the same four learners, 3 seeds)

| Scheme | CICIDS2017 macro-F1 (%) | CICIDS2017 recovery (samples) | IoTID20 macro-F1 (%) |
|---|---|---|---|
| Equal averaging (no weights) | 98.484 | 244 | 94.60 |
| PWPAE (all history; paper) | 98.495 | 260 | 94.81 |
| W-PWPAE (last 10 %, MSANA) | 98.532 | 249 | 94.48 |
| Sliding W = 100 | 98.554 | 197 | 94.31 |
| Sliding W = 500 / 1000 / 2000 | 98.556 / 98.519 / 98.514 | 263 / 268 / 249 | 94.72 / 94.78 / 94.75 |
| Fading factor α = 0.99 | **98.604** | **202** | 94.38 |
| Fading α = 0.999 / 0.9999 | 98.528 / 98.479 | 248 / 244 | 94.81 / 94.63 |
| Reset window at EDDM alarm | 98.482 | 276 | 94.75 |
| Leader-takes-all (W = 500) | 98.271 | 299 | 93.99 |
| Softmax over window error (W = 500) | 98.533 | 230 | 94.63 |
| Majority vote | 98.484 | 252 | 94.50 |

Seed std: ±0.03–0.08 pp (CIC), ±1.1–1.3 pp (IoT). **Reading:**
- On IoTID20 nothing is distinguishable from equal averaging. On CICIDS2017 the best (fading α = 0.99, sliding W = 100) are +0.07–0.12 pp macro-F1 and ≈20–25 % faster recovery than PWPAE — suggestive, about two seed-stds, and I compared 14 variants without correction, so treat it as a hypothesis, not a result.
- **PWPAE's performance weighting adds almost nothing over plain averaging** (98.97 vs 98.93 IoT; 99.144 vs 99.142 CIC accuracy). The ensemble gain comes from combining four diverse learners.
- Short/fast windows help on the stream with real drifts (CIC), long windows are neutral; leader-takes-all is worst (it throws away the averaging gain).

### 4.2 Other windows worth adding (not yet tried)

| Window | Idea | Source |
|---|---|---|
| **OASW** (sliding + adaptive window) | Sliding window monitors accuracy drop (warning → drift); an adaptive window collects new-concept samples for retraining. Same lab's lightweight alternative. | Yang & Shami, arXiv 2104.10529 |
| **Fading factor in the *detector* and in prequential evaluation** | Memoryless, faster than sliding windows; recommended for evaluation under drift. | Gama et al. 2013 |
| **ADWIN-chosen window for weighting** | Let the ADWIN window length set the weight window instead of a fixed % or fixed W. | Bifet & Gavaldà 2007 (as cited in the PWPAE/MSANA papers) |
| **KSWIN / HDDM-W / Page-Hinkley** as the drift signal for the window | KS-test window; weighted-moving-average detector; cumulative-sum detector. | River API; surveys arXiv 2004.05785, 2203.11070 |
| **Multi-scale windows (short + long)** | Weight by short- and long-window error together; react fast, stay stable. | surveys above |
| **Class-conditional windows** | Separate windows for minority/majority error so a quiet majority does not dilute minority errors (important on IoTID20). | inspired by Wang et al. 2018 (decayed class sizes) |
| **Landmark / time-based windows (tumbling, sliding, session)** | Flink offers both time and count windows; the proposal's Flink layer needs an explicit decision on which one the weighting/drift logic uses. | Flink docs (not fetched) |
| **Window for retraining after drift** | MSANA's `s = DriftArr[-1]` window; test sizes and decay. | Yang & Shami 2023 |

---

## 5. Lead's question 3 — sampling methods that might beat the reported pipeline

**Setup.** ARF-ADWIN behind each strategy, 3 seeds, same prequential protocol. Macro-F1 / MCC / balanced accuracy / FPR (rate the minority class is mis-called) in %:

| Strategy | IoTID20 macro-F1 | MCC | bal-acc | FPR(min.) | CICIDS2017 macro-F1 | MCC | bal-acc | FNR |
|---|---|---|---|---|---|---|---|---|
| None (baseline) | **91.7** | **0.841** | 87.7 | 24.4 | **97.82** | **0.956** | 97.66 | 4.01 |
| Random under 40/60 (MSANA's setting) | 77.1 | 0.585 | 88.8 | 15.3 | 94.37 | 0.890 | 96.78 | 2.99 |
| Random over 40/60 | 80.7 | 0.637 | 89.9 | 15.4 | 97.06 | 0.942 | 98.26 | 1.78 |
| Random under 50/50 | 68.1 | 0.466 | 87.6 | 10.7 | 95.98 | 0.921 | 97.35 | 3.11 |
| Random over 50/50 | 74.6 | 0.540 | 87.8 | 16.7 | 97.64 | 0.953 | **98.37** | 2.03 |
| OOB (time-decayed Poisson over-bagging) | 85.0 | 0.709 | 90.4 | 16.1 | 97.29 | 0.946 | 97.83 | 3.10 |
| UOB (under-bagging) | 70.1 | 0.490 | 88.7 | 10.9 | 77.42 | 0.600 | 86.32 | 9.18 |
| Sliding-window OOB (W = 1000) | 77.6 | 0.583 | 88.0 | 18.1 | 96.81 | 0.936 | 97.40 | 3.77 |
| Online SMOTE-style interpolation (buffer 200) | 89.6 | 0.793 | 87.4 | 24.5 | 97.33 | 0.947 | 97.36 | 4.34 |

**Findings.**
- **No resampling strategy beat "no sampling" on macro-F1 or MCC on either stream.** The authors' own MSANA IoTID20 notebook shows the same: accuracy 97.96 % without balancing vs 96.93–97.67 % with the four normalisation+resampling cells.
- Resampling does what it claims on the *minority-recall* side: balanced accuracy and G-mean rise (CIC: 97.7 → 98.3–98.4; IoT: 87.7 → up to 90.4 with OOB), minority misses fall (CIC FNR 4.0 → 1.8–2.0 %). The price is minority precision, calibration (ECE 0.02 → 0.08–0.21 on IoTID20; 0.01 → 0.02–0.16 on CICIDS2017, mild for ROS/SMOTE; Brier up to ~10×) and, for over-sampling, latency (+20–90 %) and model size.
- **Over-sampling dominates under-sampling** (consistent with MSANA choosing DROS). Under-bagging (UOB) collapses on the stream whose class ratio shifts.
- So the honest conclusion is a trade-off, not an improvement: if the operational goal is *catching attacks* (recall/balanced accuracy), ROS-50/50 or OOB; if it is *false-alarm-free precision*, no sampling.
- The default River over/under-samplers force a fixed class ratio; none of them reacts to drift. This points to the gap below.

**Sampling ideas not tried yet** (sources in brackets):
1. **Drift-triggered / severity-scaled resampling:** apply oversampling only for N samples after an alarm, with intensity ∝ S — directly the proposal's enhancement. [proposal Sec. 7; Wang et al. 2018]
2. **WEOB-style weighted combination of OOB and UOB** (OOB is more robust to changing imbalance, UOB better on static streams). [Wang, Minku & Yao 2018 / TNNLS]
3. **SMOTE variants for streams (C-SMOTE, OSMOTE):** reported to lose performance when the imbalance ratio increases; my crude SMOTE-style version was neutral. [arXiv 2204.03719]
4. **Cost-sensitive learning / threshold moving** instead of resampling (keeps calibration; tune the decision threshold on the validation stream). 
5. **Hard-example sampling** (`river.imblearn.HardSamplingClassifier`) and **instance selection for latency** (subsample the majority class after a drift so retraining is faster).
6. **k-means cluster sampling at stream level** — the PWPAE data were produced this way offline; doing it online (cluster-based reservoir) would fit the stream and cut Kafka load.

---

## 6. Current limitations of the reported work (and of what we would build on it)

**Evaluation and statistics**
1. Single runs, unseeded; no confidence intervals or significance tests. Run-to-run spread (0.2–0.6 pp on IoTID20) exceeds many published gaps (PWPAE vs MSANA 0.02–0.05 pp).
2. Headline metrics are majority-class dominated (Sec. 3.1); minority-class F1 is 84–90 % where F1 shows 99 %.
3. Drift "ground truth" is eyeballed; unmarked error bursts exist; no detection-delay or false-alarm numbers are reported.
4. PWPAE's weights use the current label before predicting (small measured effect, still a protocol flaw).
5. Latency reported as a mean on one CPU, including the learning step; no tail latency, memory or energy.
6. Ensembles of only 3 trees per learner; no ablation of leader/follower selection or of the 10 % window.

**Data**
7. 1 %-scale k-means-sampled subsets; the effect on temporal structure and drift is unquantified, and per-flow timestamps/rates are lost (relevant for a Kafka replay).
8. CICIDS2017 has documented labelling and feature-extraction errors (>25 % of flows were meaningless artefacts and >20 % of traces had to be relabelled or reconstructed in one audit) [Engelen et al. 2021]; IoTID20 has third-party-reported labelling/feature-extraction concerns, duplicates and a narrow two-device testbed [arXiv 2401.01343]. The IoTID20 class sizes in the multi-class original were not verified.
9. Binary labels only; attack-type drift and class-prior drift are collapsed. Both datasets are lab network-security captures, not physical IIoT process data (the proposal says so).

**Method / system**
10. Assumes labels arrive immediately; no unsupervised/label-delay drift handling.
11. The detectors (ADWIN, DDM, EDDM) run on the learner's *error*, so they only fire after accuracy has already fallen, and they never fired on the IoTID20 sample.
12. Reproducibility: pinned River 0.1.0, EFDT crash, notebook plotting cells fail, MSANA default config ≠ paper config.
13. Compared against 2020–2021 online learners only; no recent online-boosting/kappa-updated ensembles (e.g. KUE) or strong batch-retraining baselines.
14. **For our project:** a severity layer needs drift events to act on. IoTID20 sample has almost none; and the Kafka/Flink/River integration is unproven.

---

## 7. Pilot for the proposal's severity idea (CICIDS2017, ARF-ADWIN, 3-seed mean)

S = |E_recent − E_baseline| / (E_baseline + 0.01), E_baseline over the 300 samples before the marked drift, E_recent over the next 200.

| Marked drift at | E_baseline | E_recent | S |
|---|---|---|---|
| 9,900 | 0.000 | 0.008 | 0.83 |
| 12,600 | 0.008 | 0.010 | 0.38 |
| 14,200 | 0.007 | 0.082 | **4.50** |
| 15,800 | 0.024 | 0.045 | 0.61 |
| 18,800 | 0.006 | 0.080 | **4.82** |
| 21,600 | 0.020 | 0.092 | **2.44** |
| IoTID20: 270 / 600 | 0.019 / 0.020 | 0.027 / 0.045 | 0.35 / 0.84 |

Severity separates cleanly into low (< 1), medium (≈ 2.4) and high (> 4) on CICIDS2017, which supports the proposal's three-level design; the thresholds here are my reading of six points and would have to be tuned on a validation stream, as the proposal says. The floor ε = 0.01 matters when E_baseline ≈ 0.

---

## 8. Suggested next steps (ordered)

1. Confirm with Sidhaarth what exists for Kafka/Flink and the severity layer (Sec. 1 table).
2. Fix the evaluation protocol first: seeds, macro-F1/MCC/κ⁺, worst-window, recovery, detector delay/FAR, tail latency — code already in `experiments/analyse.py`.
3. Use CICIDS2017 as the primary stream for the severity study; decide how to get drift on IoTID20 (full dataset, injected drift, or drop H1/H3 for it).
4. Implement severity-scaled response on PWPAE (window/decay + resampling intensity by S) and compare with fading-factor weighting as a strong simple baseline.
5. Spike PyFlink + River early; if it is too heavy, a Kafka consumer in Python with Flink only for windowed aggregation is a fallback.
6. Add the interpretability outputs (Sec. 3.2 items 1–2 are cheap).

**Questions to ask Sidhaarth:** (a) What has been built for Kafka/Flink/Influx/Grafana? (b) Is the severity layer meant to act on ensemble weights, learner replacement, or sampling — or all three? (c) Are results expected on multiple seeds? (d) Which metric is the primary one for H2: F1 of label 1, macro-F1, or MCC? (e) Can we use the full datasets rather than the 1 % samples?

---

## 9. Limits of this work (what I did not verify)

- 3 seeds, one learner for the sampling study, no hyper-parameter tuning, 1 %-scale data. The window study compares **weighting windows only**.
- Search results were read at abstract/summary level; the CDDIA paper (Xu et al., IEEE TSUSC 2024) could not be located by my searches, so everything about it comes from the proposal. The 2026 arXiv papers on detector evaluation were seen through summaries.
- IoTID20 class sizes in the full dataset, Flink window semantics, and exact formulas for Kappa-M were not checked in the primary sources.

---

## 10. Sources

**Base papers and code**
- Yang, Manias, Shami. *PWPAE* (GLOBECOM 2021). https://arxiv.org/pdf/2109.05013 — code: https://github.com/Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation
- Yang, Shami. *MSANA* (IEEE TII 2023). https://arxiv.org/pdf/2210.01985 — code: https://github.com/Western-OC2-Lab/MSANA-Online-Data-Stream-Analytics-And-Concept-Drift-Adaptation
- Yang, Shami. *Lightweight concept drift detection and adaptation framework (OASW)*. https://arxiv.org/abs/2104.10529
- Xu et al. *CDDIA* (IEEE TSUSC 2024), DOI 10.1109/TSUSC.2024.3386667 — cited from the proposal only.

**Metrics and evaluation**
- Bifet, Read, Žliobaitė, Pfahringer, Holmes. *Pitfalls in benchmarking data stream classification* (ECML PKDD 2013; κ⁺). https://researchcommons.waikato.ac.nz/handle/10289/8517
- Žliobaitė, Bifet, Read, Pfahringer, Holmes. *Evaluation methods and decision theory for classification of streaming data with temporal dependence* (Mach. Learn. 2015). https://researchcommons.waikato.ac.nz/handle/10289/8954
- Gama, Sebastião, Rodrigues. *On evaluating stream learning algorithms* (Mach. Learn. 2013; fading factors). https://repositorio.inesctec.pt/handle/123456789/5355
- Brzezinski, Stefanowski. *Prequential AUC* (KAIS 2017). https://link.springer.com/article/10.1007/s10115-017-1022-8
- Chicco, Jurman. *Advantages of the MCC over F1 and accuracy* (BMC Genomics 2020). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/
- Bifet, Holmes, Pfahringer, Frank. *Fast perceptron decision tree learning from evolving data streams* (RAM-hours, PAKDD 2010) — slides: https://www.slideshare.net/slideshow/fast-perceptron-decision-tree-learning-from-evolving-data-streams/5637500 ; discussion in Haug et al. 2022, https://arxiv.org/pdf/2204.13625
- *How well does classification accuracy capture concept drift detection quality?* (MTD/MTFA/MDR overview). https://arxiv.org/pdf/2605.31186
- *A framework for evaluating and benchmarking concept drift detection methods*. https://arxiv.org/pdf/2606.07789

**Drift, windows, ensembles, imbalance**
- Learning under concept drift: a review. https://arxiv.org/pdf/2004.05785
- From concept drift to model degradation: performance-aware drift detectors. https://arxiv.org/pdf/2203.11070
- Evolving ML in non-stationary environments (drift, forgetting, adaptation). https://arxiv.org/pdf/2505.17902
- Webb et al. *Characterizing concept drift* (DMKD 2016). https://arxiv.org/abs/1511.03816
- Wang, Minku, Yao. *A systematic study of online class imbalance learning with concept drift* (IEEE TNNLS 2018). https://www.open-access.bcu.ac.uk/7721/
- *Learning from imbalanced data streams: survey and empirical study*. https://arxiv.org/pdf/2204.03719
- Cano, Krawczyk. *Kappa Updated Ensemble* (Mach. Learn. 2020). https://mlanthology.org/mlj/2020/cano2020mlj-kappa/
- Hinder et al. *Model-based explanations of concept drift*. https://arxiv.org/pdf/2303.09331v1.pdf
- Survey on semi-supervised learning for delayed partially labelled streams. https://arxiv.org/pdf/2106.09170

**Datasets**
- Engelen, Rimmer, Joosen. *Troubleshooting an intrusion detection dataset: the CICIDS2017 case study*. https://lirias.kuleuven.be/retrieve/713070 ; *Errors in the CICIDS2017 dataset…* (Lanvin et al.) https://hal.archives-ouvertes.fr/hal-03775466
- IoTGeM (re-extraction concerns on IoTID20). https://arxiv.org/pdf/2401.01343 ; Network intrusion datasets: survey and recommendations. https://arxiv.org/pdf/2502.06688

**Own artefacts (this folder):** `results/ext_summary.md` (all tables), `results/*_all.csv`, `results/fig_*.png`, `results/executed/*.ipynb` (executed authors' notebooks), `experiments/run_ext.py`, `experiments/analyse.py`, `repos/` (cloned code), `.venv/` (Python 3.8 + River 0.8.0).
