# SDA mini-project — concept drift on IoT/IIoT network streams

Goal: reproduce the PWPAE baseline, then add a drift-severity-aware response (see proposal).

| Folder | What |
|---|---|
| `baseline/` | The authors' PWPAE notebooks (IoTID20, CICIDS2017) and their sampled datasets. Unmodified. Source: [Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation](https://github.com/Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation) (MIT) |
| `results/` | `executed/` = the baseline notebooks run with outputs; `ext_summary.md`, `contrib_summary.md`, `contrib_followup.md` + `fig_*.png` = results |
| `ours/` | Our version and experiments. `severity_pwpae.py` = PWPAE plus the severity/boost/gating layers; `severity_pwpae.ipynb` = the notebook that runs and explains it; `run_contributions.py`, `more_seeds.py`, `confirm.py`, `timing_run.py`, `analyse_contrib.py` = tuning, repeated runs, fresh-seed confirmation, clean timing, paired statistics; `run_ext.py` / `analyse.py` = earlier extra experiments; `audit/` = baseline audit scripts |

## What we tested, in short

**Baseline.** PWPAE: four online models (two forest types, two drift detectors) vote on every network flow, trusting each by its past error rate. We reproduced it on CICIDS2017 and IoTID20 and then asked whether anything we add makes it recover faster after a drift. Every idea was compared with the unchanged baseline on the same data and seeds, using paired differences with confidence intervals (5-10 seeds, plus 10 fresh seeds to confirm the best one). Full process, numbers and verdicts: `ours/severity_pwpae.ipynb`.

| Idea | What it does | Verdict |
|---|---|---|
| **C1 severity-scaled reaction** | On a drift alarm, score how big the change is and react small / medium / large (re-teach models, trust recent record, replace the worst model) | No measurable gain |
| **C1b boost learning** | After an alarm, learn each new flow several times for a while | **Modest, repeatable gain**: recovery about 25 % faster, accuracy +0.075; costs extra time after each alarm |
| **C2 severity from uncertainty** | Same as C1 but the alarm and score come from model uncertainty, so no labels are needed | No gain (too many false alarms) |
| **C3 gated ensemble** | One cheap forest runs alone; the four-model ensemble wakes only after an alarm | Same accuracy at about half the time per flow |
| **Label delay test** | Labels arrive 200 flows late | Baseline's worst stretch drops from 94 % to 53 %; our reactions help only a little |
| **C4 fair evaluation** | Repeated seeds, minority-aware metrics, recovery time, tuning on a separate stretch | Shows most small differences are noise |

Also tried and dropped (no gain): resampling to balance classes, other weighting windows for the vote, other decision thresholds. Only CICIDS2017 contains drift; on IoTID20 nothing ever triggers, so our version equals the baseline there.

## Run
The authors' River 0.1.0 no longer installs; use Python 3.8 + River 0.8.0.
```bash
uv venv --python 3.8 .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python.exe -m ipykernel install --sys-prefix --name python3
cd baseline && ../.venv/Scripts/python.exe -m nbconvert --to notebook --execute --allow-errors globecom2021_PWPAE_CICIDS2017.ipynb --output-dir ../results/executed && cd ..
.venv/Scripts/python.exe ours/run_ext.py 4     # worker count; lower it on MemoryError
.venv/Scripts/python.exe ours/analyse.py       # -> results/ext_summary.md, figures
```
`--allow-errors` is needed: the EFDT cell crashes under River 0.8.0 and the final plot cell then fails.

Our version (`ours/severity_pwpae.ipynb` runs one seed live and shows the saved results):
```bash
.venv/Scripts/python.exe ours/run_contributions.py tune 4   # thresholds on a validation stretch
.venv/Scripts/python.exe ours/run_contributions.py eval 4   # all variants, then: ours/analyse_contrib.py
.venv/Scripts/python.exe ours/more_seeds.py run 4           # seeds 1-10 incl. boost; then: ours/confirm.py run 4
```
