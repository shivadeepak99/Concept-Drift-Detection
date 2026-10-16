# Follow-up results (collected from ours/more_seeds.py report and ours/confirm.py report)

The first round (`contrib_summary.md`) showed that the severity-scaled reaction C1 changed nothing measurable, and that replaying old flows does little because the models already learn every flow. So we added a reaction that **boosts learning on the new flows** after an alarm (`act_boost`: medium = each new flow learned 2x for 500 flows, high = 4x for 1,000 flows) and compared on 10 seeds (CICIDS2017, scored on the last 70 % of the stream).

Variants: `C1` = severity-scaled, old reaction | `C1_uniform_high` = old reaction at maximum strength on every alarm | `C1b_boost` = severity-scaled, with boost | `C1b_uniform_high` = maximum reaction + boost on every alarm | `C1b_boost_only` = only the boost | `C1b_boost_fasttrust` = boost + fast-trust weights.
Caution: `C1b_uniform_high` was picked *after* seeing these ten seeds, so it is confirmed again on fresh seeds below.

## Seeds 1-10, paired difference to the baseline
```
paired difference to the baseline over 10 seeds (mean ±95% CI; * = excludes 0). Positive = higher.

C1
  acc             -0.001 ±0.049   (seeds better: 3/10)
  macro_f1        -0.002 ±0.097   (seeds better: 3/10)
  mcc             -0.000 ±0.002   (seeds better: 3/10)
  worst_acc_w500  +0.140 ±0.661   (seeds better: 5/10)
  drop            -0.328 ±0.599   (seeds better: 5/10)
  recovery        -7.850 ±25.218   (seeds better: 6/10)

C1_uniform_high
  acc             +0.014 ±0.045   (seeds better: 6/10)
  macro_f1        +0.027 ±0.089   (seeds better: 6/10)
  mcc             +0.001 ±0.002   (seeds better: 6/10)
  worst_acc_w500  +0.360 ±0.817   (seeds better: 4/10)
  drop            -0.483 ±0.784   (seeds better: 5/10)
  recovery        -31.483 ±41.814   (seeds better: 6/10)

C1b_boost
  acc             +0.029 ±0.032   (seeds better: 6/10)
  macro_f1        +0.055 ±0.064   (seeds better: 6/10)
  mcc             +0.001 ±0.001   (seeds better: 6/10)
  worst_acc_w500  +0.620 ±0.488 *   (seeds better: 7/10)
  drop            -0.256 ±0.559   (seeds better: 7/10)
  recovery        -39.533 ±34.461 *   (seeds better: 6/10)

C1b_uniform_high
  acc             +0.107 ±0.071 *   (seeds better: 8/10)
  macro_f1        +0.212 ±0.141 *   (seeds better: 8/10)
  mcc             +0.004 ±0.003 *   (seeds better: 8/10)
  worst_acc_w500  +1.100 ±0.630 *   (seeds better: 8/10)
  drop            -0.661 ±0.770   (seeds better: 7/10)
  recovery        -62.683 ±41.327 *   (seeds better: 9/10)

C1b_boost_only
  acc             +0.070 ±0.060 *   (seeds better: 8/10)
  macro_f1        +0.138 ±0.118 *   (seeds better: 8/10)
  mcc             +0.003 ±0.002 *   (seeds better: 8/10)
  worst_acc_w500  +1.000 ±0.815 *   (seeds better: 7/10)
  drop            -0.572 ±0.733   (seeds better: 8/10)
  recovery        -35.783 ±29.051 *   (seeds better: 8/10)

C1b_boost_fasttrust
  acc             +0.079 ±0.065 *   (seeds better: 9/10)
  macro_f1        +0.154 ±0.128 *   (seeds better: 9/10)
  mcc             +0.003 ±0.003 *   (seeds better: 9/10)
  worst_acc_w500  +1.080 ±0.942 *   (seeds better: 6/10)
  drop            -0.600 ±0.971   (seeds better: 7/10)
  recovery        -29.400 ±54.996   (seeds better: 7/10)
B0 recovery mean 250 ± 37
C1 recovery mean 242 ± 32
C1_uniform_high recovery mean 218 ± 39
C1b_boost recovery mean 210 ± 34
C1b_uniform_high recovery mean 187 ± 33
C1b_boost_only recovery mean 214 ± 29
C1b_boost_fasttrust recovery mean 220 ± 50
```

## Confirmation on fresh seeds (21-30), no-harm check on IoTID20, and labels 200 flows late
```

CICIDS2017, fresh seeds 21-30: always-react-hard + boost  vs  baseline  (n=10 paired seeds; mean difference ±95% CI, * = excludes 0)
  acc             +0.075 ±0.052 *   better in 9/10 seeds
  macro_f1        +0.147 ±0.101 *   better in 9/10 seeds
  mcc             +0.003 ±0.002 *   better in 9/10 seeds
  worst_acc_w500  +0.700 ±0.743   better in 9/10 seeds
  drop            -0.433 ±0.725   better in 7/10 seeds
  recovery        -61.783 ±38.584 *   better in 9/10 seeds

IoTID20, seeds 1-3: same variant vs baseline  (n=3 paired seeds; mean difference ±95% CI, * = excludes 0)
  acc             +0.042 ±0.298   better in 1/3 seeds
  macro_f1        +0.250 ±1.630   better in 1/3 seeds
  mcc             +0.004 ±0.031   better in 1/3 seeds
  worst_acc_w500  +0.200 ±1.791   better in 1/3 seeds
  drop            +nan ±nan   better in 3/3 seeds
  recovery        +nan ±nan   better in 3/3 seeds

CICIDS2017, labels 200 flows late, seeds 1-5: same variant vs baseline with the same delay  (n=5 paired seeds; mean difference ±95% CI, * = excludes 0)
  acc             +0.224 ±0.067 *   better in 5/5 seeds
  macro_f1        +0.480 ±0.138 *   better in 5/5 seeds
  mcc             +0.009 ±0.003 *   better in 5/5 seeds
  worst_acc_w500  +6.280 ±3.715 *   better in 5/5 seeds
  drop            -0.644 ±3.391   better in 2/5 seeds
  recovery        -28.833 ±45.898   better in 4/5 seeds
```
