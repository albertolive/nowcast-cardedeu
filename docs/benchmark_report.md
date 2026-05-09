# Benchmark vs baselines

> ⚠️ **PRELIMINARY** — sample size is small. The headline finding is that there are only **43 rain events across 4 distinct days** in the verified log. Confidence intervals below are wide; any 'winner' claim with overlapping CIs is not supported.

- Source log: `data/predictions_log.jsonl`
- Verified predictions: **3788** (positives: **43**)
- Predictions with both AEMET and ensemble baselines: **1660** (positives: **39**)
- Strict anticipation subset (station dry + radar > 15 km): **1650** (positives: **39**)

Metrics:
- **Brier ↓**: mean squared error between predicted prob and outcome. Lower is better.
- **LogLoss ↓**: penalises confident mistakes harder. Lower is better.
- **AUC ↑**: rank-discrimination. 0.5 = chance, 1.0 = perfect ordering.
- **Acc@τ**: accuracy when the predictor is thresholded at τ. Sensitive to base rate at low N.
- **Calibration**: predicted prob within a bin should match observed frequency.

All intervals are 95% bootstrap CIs over the per-row sample (1000 iterations).

## Global — local on full sample

N=3788, positives=43

| predictor | N | N+ | base | Brier ↓ | LogLoss ↓ | AUC ↑ | Acc@0.5 | Acc@0.65 |
|---|---|---|---|---|---|---|---|---|
| Local (XGBoost) | 3788 | 43 | 0.011 | 0.0209 [0.0181, 0.0236] | 0.0810 [0.0729, 0.0894] | 0.9083 [0.8616, 0.9460] | 0.9691 [0.9638, 0.9744] | 0.9839 [0.9799, 0.9879] |
| AEMET prob_precip | 1660 | 39 | 0.023 | 0.3497 [0.3262, 0.3713] | 7.6273 [7.0215, 8.2191] | 0.5937 [0.5107, 0.6775] | 0.6361 [0.6127, 0.6608] | 0.6488 [0.6253, 0.6723] |
| Ensemble agreement (4 NWP) | 1746 | 39 | 0.022 | 0.1322 [0.1214, 0.1436] | 1.4621 [1.2181, 1.7249] | 0.8953 [0.8529, 0.9386] | 0.7377 [0.7176, 0.7583] | 0.8900 [0.8746, 0.9044] |

### Local (XGBoost) — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 3539 | 0.027 | 0.005 |
| 0.20–0.40 | 96 | 0.293 | 0.021 |
| 0.40–0.60 | 106 | 0.515 | 0.142 |
| 0.60–0.80 | 34 | 0.673 | 0.147 |
| 0.80–1.00 | 13 | 0.800 | 0.385 |

### AEMET prob_precip — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 867 | 0.004 | 0.025 |
| 0.20–0.40 | 101 | 0.273 | 0.000 |
| 0.40–0.60 | 107 | 0.431 | 0.000 |
| 0.60–0.80 | 32 | 0.680 | 0.000 |
| 0.80–1.00 | 553 | 0.982 | 0.031 |

### Ensemble agreement (4 NWP) — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 884 | 0.000 | 0.000 |
| 0.20–0.40 | 379 | 0.250 | 0.018 |
| 0.40–0.60 | 274 | 0.500 | 0.015 |
| 0.60–0.80 | 107 | 0.750 | 0.019 |
| 0.80–1.00 | 102 | 1.000 | 0.255 |


## Apples-to-apples — entries with all three predictors

N=1660, positives=39

| predictor | N | N+ | base | Brier ↓ | LogLoss ↓ | AUC ↑ | Acc@0.5 | Acc@0.65 |
|---|---|---|---|---|---|---|---|---|
| Local (XGBoost) | 1660 | 39 | 0.023 | 0.0387 [0.0334, 0.0442] | 0.1364 [0.1221, 0.1508] | 0.9031 [0.8694, 0.9349] | 0.9404 [0.9283, 0.9518] | 0.9699 [0.9620, 0.9783] |
| AEMET prob_precip | 1660 | 39 | 0.023 | 0.3497 [0.3262, 0.3713] | 7.6273 [7.0215, 8.2191] | 0.5937 [0.5107, 0.6775] | 0.6361 [0.6127, 0.6608] | 0.6488 [0.6253, 0.6723] |
| Ensemble agreement (4 NWP) | 1660 | 39 | 0.023 | 0.1356 [0.1233, 0.1474] | 1.4961 [1.2241, 1.7767] | 0.8933 [0.8486, 0.9358] | 0.7301 [0.7084, 0.7512] | 0.8855 [0.8711, 0.9012] |

### Local (XGBoost) — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 1447 | 0.041 | 0.008 |
| 0.20–0.40 | 75 | 0.301 | 0.027 |
| 0.40–0.60 | 104 | 0.515 | 0.144 |
| 0.60–0.80 | 24 | 0.680 | 0.208 |
| 0.80–1.00 | 10 | 0.800 | 0.500 |

### AEMET prob_precip — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 867 | 0.004 | 0.025 |
| 0.20–0.40 | 101 | 0.273 | 0.000 |
| 0.40–0.60 | 107 | 0.431 | 0.000 |
| 0.60–0.80 | 32 | 0.680 | 0.000 |
| 0.80–1.00 | 553 | 0.982 | 0.031 |

### Ensemble agreement (4 NWP) — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 837 | 0.000 | 0.000 |
| 0.20–0.40 | 350 | 0.250 | 0.020 |
| 0.40–0.60 | 266 | 0.500 | 0.015 |
| 0.60–0.80 | 107 | 0.750 | 0.019 |
| 0.80–1.00 | 100 | 1.000 | 0.260 |


## Anticipation only — station dry & radar > 15 km

N=1650, positives=39

| predictor | N | N+ | base | Brier ↓ | LogLoss ↓ | AUC ↑ | Acc@0.5 | Acc@0.65 |
|---|---|---|---|---|---|---|---|---|
| Local (XGBoost) | 1650 | 39 | 0.024 | 0.0380 [0.0330, 0.0433] | 0.1345 [0.1208, 0.1491] | 0.9046 [0.8712, 0.9338] | 0.9412 [0.9297, 0.9521] | 0.9703 [0.9624, 0.9788] |
| AEMET prob_precip | 1650 | 39 | 0.024 | 0.3463 [0.3240, 0.3687] | 7.5375 [6.9551, 8.1125] | 0.5958 [0.5166, 0.6823] | 0.6394 [0.6164, 0.6642] | 0.6521 [0.6291, 0.6764] |
| Ensemble agreement (4 NWP) | 1650 | 39 | 0.024 | 0.1343 [0.1232, 0.1462] | 1.4998 [1.2278, 1.7914] | 0.8940 [0.8459, 0.9355] | 0.7327 [0.7109, 0.7539] | 0.8879 [0.8733, 0.9030] |

### Local (XGBoost) — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 1443 | 0.041 | 0.008 |
| 0.20–0.40 | 74 | 0.301 | 0.027 |
| 0.40–0.60 | 100 | 0.516 | 0.150 |
| 0.60–0.80 | 23 | 0.680 | 0.217 |
| 0.80–1.00 | 10 | 0.800 | 0.500 |

### AEMET prob_precip — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 867 | 0.004 | 0.025 |
| 0.20–0.40 | 100 | 0.272 | 0.000 |
| 0.40–0.60 | 107 | 0.431 | 0.000 |
| 0.60–0.80 | 32 | 0.680 | 0.000 |
| 0.80–1.00 | 544 | 0.982 | 0.031 |

### Ensemble agreement (4 NWP) — calibration
| bin | N | predicted mean | observed freq |
|---|---|---|---|
| 0.00–0.20 | 836 | 0.000 | 0.000 |
| 0.20–0.40 | 348 | 0.250 | 0.020 |
| 0.40–0.60 | 264 | 0.500 | 0.015 |
| 0.60–0.80 | 102 | 0.750 | 0.020 |
| 0.80–1.00 | 100 | 1.000 | 0.260 |


## How to read this

If the local model's Brier CI fully sits below the baselines' CIs, the local model is calibrated better. If they overlap (which is likely at this sample size), conclude **inconclusive** — the data isn't there yet.

AEMET and ensemble predictions are only present when the rain gate was open (see `predict.py:265-275`). The full-baseline subset therefore over-represents moments where rain was already plausible — a selection bias inflating all predictors' apparent accuracy. Treat absolute numbers cautiously and focus on relative ordering.
