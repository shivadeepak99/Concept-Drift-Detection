Deeper audit of the PWPAE baseline (run from the repo root, after `ours/run_ext.py`):

| Script | Question |
|---|---|
| `1_data_and_error_structure.py` | How are attacks/errors laid out in the stream? How different are the 4 models? How much is the ensemble worth, and where? |
| `2_duplicates_early_warning_gating.py` | Repeated rows, input-feature drift alarms, "ensemble only after an alarm" estimate |
| `3_threshold_and_alarm_precision.py` | Is the 0.5 decision threshold leaving anything on the table? Do input-drift alarms predict error bursts? |
| `4a_...label_delay.py` / `4b_compare_...py` | Does a plain 10-tree forest rival PWPAE? How much does delaying the labels hurt? |
| `5_uncertainty_vs_error.py` | Can model uncertainty (no labels) stand in for the error rate? |
