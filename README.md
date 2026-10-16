# SDA mini-project — PWPAE / MSANA concept-drift implementations

Code for the IOE413 Streaming Data Analytics project *Adaptive Concept Drift Detection and Response for Real-Time IIoT Network Data Streams*.
Holds the notebooks the group lead asked us to run, plus extension experiments (extra metrics, window schemes, sampling strategies) and the notes document.

| Path | What |
|---|---|
| `pwpae/` | Authors' PWPAE notebooks (IoTID20, CICIDS2017) + sampled datasets. Source: [Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation](https://github.com/Western-OC2-Lab/PWPAE-Concept-Drift-Detection-and-Adaptation) (MIT) |
| `msana/` | Authors' MSANA notebooks (`TII-MSANA-CICIDS2017.ipynb`, `TII-MSANA-IoTID20.ipynb`) + data. Source: [Western-OC2-Lab/MSANA-Online-Data-Stream-Analytics-And-Concept-Drift-Adaptation](https://github.com/Western-OC2-Lab/MSANA-Online-Data-Stream-Analytics-And-Concept-Drift-Adaptation) (MIT) |
| `experiments/run_ext.py` | Re-runs the 4 PWPAE learners and ARF-ADWIN under 9 sampling strategies, 3 seeds, both datasets (~2 min on 16 workers, ~10 min on 4) |
| `experiments/analyse.py` | Metrics, ensemble-window schemes, drift-detector study, figures → `results/` |
| `results/executed/` | The four authors' notebooks, executed with outputs (River 0.8.0) |
| `results/ext_summary.md`, `*_all.csv`, `fig_*.png` | Extension results |
| `SDA_Project_Notes.md` / `.docx` | Status, reproduction, findings and sources |

The notebooks are unmodified copies of the authors' code (MIT, see `LICENSE` in each folder).

## Run

The authors pinned River 0.1.0 (Python 3.6); it is not installable on current systems. Use Python 3.8 + River 0.8.0:

```bash
uv venv --python 3.8 .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python.exe -m ipykernel install --sys-prefix --name python3

# authors' notebooks (run inside their folder; they read ./data or Data/)
cd pwpae && ../.venv/Scripts/python.exe -m nbconvert --to notebook --execute --allow-errors globecom2021_PWPAE_CICIDS2017.ipynb --output-dir ../results/executed
cd ../msana && ../.venv/Scripts/python.exe -m nbconvert --to notebook --execute --allow-errors TII-MSANA-CICIDS2017.ipynb --output-dir ../results/executed

# extension experiments (from the repo root)
.venv/Scripts/python.exe experiments/run_ext.py 4      # arg = worker processes (each holds a model; lower it if you hit MemoryError / paging-file errors)
.venv/Scripts/python.exe experiments/analyse.py        # writes results/ext_summary.md, CSVs, figures
```

## Known issues (see notes §2)
- `ExtremelyFastDecisionTreeClassifier` raises `AttributeError: ... 'merit'` under River 0.8.0, so the EFDT cell fails and the final comparison-plot cell raises `NameError` (hence `--allow-errors`).
- Models are unseeded in the authors' notebooks; results vary run to run.
- The MSANA CICIDS2017 file has 14,155 rows; the paper text says 28,303.
