# SDA mini-project — concept drift on IoT/IIoT network streams

Goal: reproduce the PWPAE baseline, then add a drift-severity-aware response (see proposal).

| Folder | What |
|---|---|
| `baseline/` | The authors' PWPAE notebooks (IoTID20, CICIDS2017) and their sampled datasets. Unmodified. Source: [Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation](https://github.com/Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation) (MIT) |
| `results/` | `executed/` = the baseline notebooks run with outputs; `ext_summary.md` + `fig_*.png` = results of our extra experiments |
| `ours/` | Our work so far: `run_ext.py` (runs the baseline learners + sampling strategies, 3 seeds) and `analyse.py` (extra metrics, window schemes, drift detectors). **The severity-aware version is not built yet.** |

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
