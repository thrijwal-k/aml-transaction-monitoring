# Transaction Monitoring & Fraud Detection

A transaction monitoring system for mobile money payments. It combines the two approaches banks use to find financial crime:

1. **Rule-based scenarios**: the transparent first line of detection that compliance teams can explain to a regulator.
2. **Machine learning (XGBoost)**: learns patterns the rules miss and ranks alerts by risk.

Every alert comes with plain-English **reasons**, so an investigator can see *why* a transaction was flagged, plus a **Streamlit dashboard** with the investigator queue, scenario performance and model diagnostics.

## Why this matters

Transaction monitoring teams face two problems at once: missed fraud, and too many false positives for investigators to review. This project measures both. It compares the existing flag, rules on their own, machine learning on its own, and a hybrid, using recall, precision, false-positive rate and alert volume.

## Data

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Lopez-Rojas et al., 2016) is a synthetic dataset of 6.36 million mobile money transactions over 30 days, simulated from real transaction logs. It has 8,213 fraud cases (0.13%) in which fraudsters take over accounts, empty them with a **transfer** to a mule account, and **cash out**.

Download it free from Kaggle (you need a Kaggle account) and save the CSV as `data/paysim.csv`.

## Approach

| Step | What it does |
|---|---|
| **Scope** | Monitors TRANSFER and CASH_OUT, the only types where fraud occurs (2.77m transactions) |
| **Features** | Amount, amount relative to balance, account emptied, balance reconciliation checks, rapid pass-through (the same amount transferred and cashed out within an hour), receiver fan-in and daily activity |
| **Rules (R1–R5)** | Large transfer over 200,000 · account emptied · rapid pass-through · receiver balance mismatch · receiver gets money from 3+ senders in a day |
| **Model** | XGBoost with class weighting for 0.13% fraud. **Time-based split**: trained on days 1–20, tested on days 21–31, so the test never uses future data. The threshold is tuned on the last 4 training days to maximise F2 (recall counts double, because a missed fraud costs more than an extra review). |
| **Comparison** | Isolation Forest, an unsupervised model trained without labels, shows what is possible when no fraud labels exist |
| **Hybrid alerting** | Alert if the model score passes the threshold **or** 3+ rules fire |
| **Explainability** | Each alert's top 3 reasons come from XGBoost's built-in SHAP contributions |
| **Stress test** | PaySim fraud is unusually easy to separate: balance fields are inconsistent only for fraud, fraud transfers and cash-outs come in exactly matched pairs, and the final days are fraud-heavy. The model is retrained with each group of artefacts removed, and precision is re-weighted to the monthly fraud rate, to show what holds up |

## Results

Run the pipeline, then `python report.py`. It writes `RESULTS.md` with the full comparison table.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py --data data/paysim.csv   # about 1-3 minutes and 3-4 GB RAM on a laptop
python report.py                                # writes RESULTS.md and prints the headline figures
streamlit run app.py                            # opens the dashboard in your browser
```

No data yet? `python run_pipeline.py --synthetic` runs everything on generated test data. Those numbers are for testing only.

Tests: `python -m pytest -q`

## Project structure

```
run_pipeline.py      end-to-end pipeline: features, rules, models, evaluation, alerts
report.py            turns the metrics into RESULTS.md
app.py               Streamlit alerts dashboard
src/data.py          PaySim loader and synthetic test-data generator
src/features.py      transaction features and plain-English reason labels
src/rules.py         monitoring scenarios R1-R5 and per-rule performance
tests/               unit tests for rules and features
```

## Limitations and next steps

- PaySim is simulated. Real transaction monitoring adds customer due diligence (KYC) data, sanctions screening and account history.
- The receiver fan-in rule works best on network-style data. A graph approach (for example, linking accounts that share mules) would extend it.
- Rule thresholds are fixed. In practice, they would be tuned by "above and below the line" testing with investigators.
