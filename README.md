# SDA mini-project — concept drift on IoT/IIoT network streams

Goal: reproduce the PWPAE baseline, then add a drift-severity-aware response (see proposal).

| Folder | What |
|---|---|
| `baseline/` | The authors' PWPAE notebooks (IoTID20, CICIDS2017) and their sampled datasets. Unmodified. Source: [Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation](https://github.com/Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation) (MIT) |
| `results/` | `executed/` = the baseline notebooks run with outputs; `ext_summary.md`, `contrib_summary.md`, `contrib_followup.md` + `fig_*.png` = results |
| `ours/` | Our version and experiments. `severity_pwpae.py` = PWPAE plus the severity/boost/gating layers; `severity_pwpae.ipynb` = the notebook that runs and explains it; `run_contributions.py`, `more_seeds.py`, `confirm.py`, `timing_run.py`, `analyse_contrib.py` = tuning, repeated runs, fresh-seed confirmation, clean timing, paired statistics; `run_ext.py` / `analyse.py` = earlier extra experiments; `audit/` = baseline audit scripts |

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
