# DrugWorld D4: validation gates, baseline strength and checklist (2026-11-02)

Branch `drugworld` of private `rafidquadir-eng/DrugWorld`. The design commit is `eefeeb0` (A1). The D2 freeze is
`1174ae0`. Selection was committed at `9bf991e`, before any evaluation output. Every number below
comes from `results/d4/gates_val.json`, `results/d4/controls_val.json` and
`results/d4/checklist*.json`. These are **validation** outcomes only. No test condition has been
forecast or scored. **Stopped for the author's "go".**

## 1. What ran

- **D3:** 360 training runs with 0 failures. The grid has 312 runs: 2 widths × 2 dropouts × 2 weight decays × 3
  seeds per family and set, plus 7 H2b families. The after stage has 48 runs: config-matched references,
  `dw_line_k` (4 lines × 3 seeds) and global-normalisation runs. Every checkpoint is backed up with SHA-256.
- **ODE fit:** not run. F-Decode failed for idr0119 (log-count raw R² 0.798 < 0.8), so A1 O1
  drops the idr0119 decoded readouts and the ODE comparisons. G-Combo+ and G-Plan (b) are
  therefore not testable.
- **D4:** forecasts for 114 run sets on 8 GPUs, then gates and controls in parallel, then the checklist.
- **Two evaluation-code bugs, fixed in `673356d` before these files were committed.** Neither
  changes a rule.
  1. H2b in the global pass crashed on a missing config-matched reference. The fix skips it, as
     H1 and H2a already did, and an incomplete comparison set cannot pass.
  2. The H1c line-offset anchor was NaN. The held-out line's vehicle wells were not counted as
     context. The plan's definition, "added to the held-out line's vehicle", is now implemented.

## 2. Gates on validation (primary normalisation, primary horizons)

Errors are mean unit errors (lower is better). "rel" is full/other − 1. The p-values are one-sided
sign flips with 100k draws.

| gate | outcome | detail |
|---|---|---|
| **G-Dose** (H1, 29 units, 399 fields) | **fail** | beats `dw_none` matched (1.174 vs 1.555, rel −0.245, p 0.0013) and `dw_doseshuf` matched (1.174 vs 1.379, rel −0.149, p 9e-5); **does not beat nearest dose** (1.174 vs 1.137, rel +0.033, p 0.77) |
| **G-Dose+** | **fail** | requires G-Dose; linear interpolation 1.140 (rel +0.030, p 0.78) |
| **G-Combo** (H2a, 7 units, 84 fields; val pair Gemcitabine+Lapatinib) | **pass** | beats HSA (1.037 vs 1.151, rel −0.099, p 0.008) and additive (1.037 vs 1.239, rel −0.163, p 0.008); with 7 units, 0.008 is the smallest attainable p (7/7 better) |
| **G-Combo+** | not testable | F-Decode failed for idr0119 (A1 O1) |
| **G-Compound** (H2b, 41 compounds) | **fail** | no source beats all three references; `dw_none` is better than `dw_full` for every source (ecfp4 1.107 vs 0.950; molformer 1.064 vs 0.940; target 1.049 vs 0.937); Holm p = 1.0 for each source |
| **G-Plan (a)** (dose inversion, 100 wells) | **fail** | median \|log10 dose error\|: model 0.199, nearest 0.398, interpolation 0.300; beats nearest (p 2.5e-4) but **not interpolation** (p 0.30) |
| **G-Plan (b)** | not testable | needs the decoded 72 h cell count (F-Decode) |
| **H1c** (fold MDAMB157, 18 units; 147 fields excluded under A1 C5) | **fail** | `dw_line` 2.157 vs line offset 1.986 (rel +0.086, p 0.9999) and vs lines mean 2.045 (p 0.9999) |

Also reported:
- H1 `dw_full` beats `dw_doseonly` (rel −0.21, p 3.6e-4).
- Own-configuration references give the same picture as the matched ones.
- H2b per source:
  - ecfp4 loses to `dw_structshuf` (rel +0.013) and kNN (+0.032);
  - molformer is level with `dw_structshuf` (+0.005, p 0.62) and kNN (−0.009, p 0.34);
  - target beats `dw_structshuf` (rel −0.070, p 0.0019) but not kNN (−0.023, p 0.12).

**Global normalisation (sensitivity):** the same pass/fail pattern. G-Combo passes (HSA rel −0.087, additive rel
−0.150, p 0.008 each); G-Dose, G-Dose+ and G-Compound fail. The global runs exist only at each family's own
configuration, so the config-matched comparisons are absent there. H1c has no global runs.

## 3. Baseline strength (L3) and substitutions

Every lookup baseline beats copy-last on validation, so **none is weak and no substitution
applies**. The test will use nearest dose, linear interpolation, HSA, additive and kNN as written.

| set | baseline | vs copy-last |
|---|---|---|
| H1 | nearest dose | 1.137 vs 1.923, rel −0.41, p 1e-5 |
| H1 | interpolation | 1.140 vs 1.923, rel −0.41, p 1e-5 |
| H1 | residual (λ 1.0) | rel −0.41, p 1e-5 |
| H2a | HSA | 1.150 vs 1.450, rel −0.21, p 0.008 |
| H2a | additive | 1.239 vs 1.450, rel −0.15, p 0.008 |
| H2b | kNN | 1.073 vs 1.495, rel −0.28, p 1e-5 |
| H2b | residual (λ 0.0) | 0.937 vs 1.495, rel −0.37, p 1e-5 |

H2a baselines used the source preference recorded in the protocol (non-re-dosed singles first). The source for each
component dose is in `gates_val.json` under `baseline_single_sources`.

## 4. Controls (reported; never used to change a gate)

- **Static clip:** model/copy-last error ratio 0.57 (H1), 0.70 (H2a) and 0.65–0.69 (H2b) on
  measured targets, against 0.92–0.96 on static clips. The models use the dynamics.
- **Context swap:** the error rises from 1.04 to 2.05 (H1), 0.69 to 1.65 (H2a) and 0.63–0.70 to 1.38–1.55 (H2b).
- **Action shuffle:** the error rises by 3–10%: H1 1.095 → 1.169; H2a 1.013 → 1.046; H2b ecfp4 1.036 → 1.113,
  molformer 0.981 → 1.054, target 0.965 → 1.066.
- **Leakage:**
  - 0 frame-0 hash duplicates across roles in either set;
  - 0 idr0119 fingerprint pairs with r > 0.98 across roles; **2 idr0064 pairs**;
  - 0 parent InChIKeys in two splits;
  - 0 held-out scaffolds in training.
- **Plate-batch probe:** replicate plates are identified with mean accuracy 0.967, against chance 0.284
  (32 conditions). There is a strong plate signature in the latents, as in GeneWorld.

## 5. D4 checklist (plan section 10): 9/9 pass

| # | item | result |
|---|---|---|
| 1 | commit order: freeze → selection → evaluation outputs → `gates_val.json` | PASS (re-checked on HEAD with `--final`) |
| 2 | 3 seeds per family; config-matched references present | PASS |
| 3 | no learned ID embedding in any gated model | PASS |
| 4 | permutation hashes for shuffled families | PASS |
| 5 | evaluated unit counts = frozen counts − listed exclusions | PASS |
| 6 | baseline-strength table and implied substitutions | PASS (none) |
| 7 | no file refers to test conditions | PASS |
| 8 | primary-normalised and global results both present | PASS |
| 9 | feasibility-gate results recorded | PASS |

## 6. Points for the author

1. On validation, only G-Combo passes. The test runs once after "go", and the claim rules apply
   mechanically. Validation does not change the test design.
2. **G-Combo rests on 7 validation units.** The test has 4 held-out pairs (about 30 fields).
3. **H2b `dw_none` beats every structure-conditioned model on validation.** Structure
   information has not helped here.
4. **The decoder α sits at the grid edge** (1e5 for all idr0119 targets). This was reported at the freeze and
   not changed.
5. **Cost:** the 8 × A100-40GB instance runs at about $10.3/h; please confirm the rate on the console. D3 plus D4 took
   about 19 h of wall time, roughly $195. D5 should take about 15 minutes of GPU time (test forecasts), plus
   CPU for gates and controls. **Nothing is running now.**
