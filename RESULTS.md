# Results

Data: PaySim, 6,362,620 transactions (8,213 fraud).
Evaluated on 134,160 unseen TRANSFER/CASH_OUT transactions containing 2,870 fraud cases.
Split: train steps < 480 (threshold tuned on steps 384-479), test steps >= 480.

| Strategy | Alerts | Fraud caught | Recall | Precision | False-positive rate |
|---|---|---|---|---|---|
| Existing PaySim flag | 10 | 10 | 0.3% | 100.0% | 0.000% |
| Rules: 2+ scenarios | 12,395 | 2,860 | 99.7% | 23.1% | 7.263% |
| Rules: 3+ scenarios | 1,487 | 1,427 | 49.7% | 96.0% | 0.046% |
| XGBoost | 2,868 | 2,868 | 99.9% | 100.0% | 0.000% |
| Hybrid: XGBoost or 3+ rules | 2,928 | 2,868 | 99.9% | 98.0% | 0.046% |

Precision-recall AUC: XGBoost 1.000, XGBoost without balance fields 0.998,
Isolation Forest (unsupervised) 0.990. ROC-AUC (XGBoost) 1.000.

## Stress test: removing simulator artefacts

PaySim fraud is unusually easy to separate: balance fields are inconsistent only for fraud, and fraud transfers and
cash-outs come in exactly matched pairs. The final days are also fraud-heavy (2.14% of test
transactions, compared with 0.30% across the month), which inflates precision. Each model below
is retrained without one group of artefacts. The last column re-weights precision to the monthly fraud rate.

| Features used | PR-AUC | Recall | Precision | Precision at monthly fraud rate |
|---|---|---|---|---|
| All features | 1.000 | 99.9% | 100.0% | 100.0% |
| Without balance fields | 0.998 | 99.4% | 99.9% | 99.0% |
| Without balance fields or pass-through pairing | 0.407 | 42.6% | 48.1% | 11.2% |
| Behaviour only (amount, type, receiver activity) | 0.319 | 24.9% | 58.6% | 16.2% |
