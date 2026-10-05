"""Turn outputs/metrics.json into RESULTS.md and print CV-ready figures. Run after run_pipeline.py."""
import json
from pathlib import Path

m = json.loads(Path("outputs/metrics.json").read_text())
s = m["strategies"]
h, x, base = s["hybrid_xgboost_or_3plus_rules"], s["xgboost"], s["existing_paysim_flag"]
r2 = s["rules_2plus"]
pa = m["pr_auc"]

rows = "\n".join(
    f"| {name} | {v['alerts']:,} | {v['true_positives']:,} | {v['recall']:.1%} | {v['precision']:.1%} | "
    f"{v['false_positive_rate']:.3%} |"
    for name, v in [("Existing PaySim flag", base), ("Rules: 2+ scenarios", r2),
                    ("Rules: 3+ scenarios", s["rules_3plus"]), ("XGBoost", x), ("Hybrid: XGBoost or 3+ rules", h)]
)
md = f"""# Results

Data: {m['data_source']}, {m['transactions_total']:,} transactions ({m['fraud_total']:,} fraud).
Evaluated on {m['test_transactions']:,} unseen TRANSFER/CASH_OUT transactions containing {m['test_fraud']:,} fraud cases.
Split: {m['split']}.

| Strategy | Alerts | Fraud caught | Recall | Precision | False-positive rate |
|---|---|---|---|---|---|
{rows}

Precision-recall AUC: XGBoost {pa['xgboost']:.3f}, XGBoost without balance fields {pa['xgboost_without_balance_fields']:.3f},
Isolation Forest (unsupervised) {pa['isolation_forest']:.3f}. ROC-AUC (XGBoost) {m['roc_auc_xgboost']:.3f}.
"""
st = m["stress_test"]
md += f"""
## Stress test: removing simulator artefacts

PaySim fraud is unusually easy to separate: balance fields are inconsistent only for fraud, and fraud transfers and
cash-outs come in exactly matched pairs. The final days are also fraud-heavy ({m['fraud_rate_test_period']:.2%} of test
transactions, compared with {m['fraud_rate_monitored_month']:.2%} across the month), which inflates precision. Each model below
is retrained without one group of artefacts. The last column re-weights precision to the monthly fraud rate.

| Features used | PR-AUC | Recall | Precision | Precision at monthly fraud rate |
|---|---|---|---|---|
""" + "\n".join(f"| {r['features']} | {r['pr_auc']:.3f} | {r['recall']:.1%} | {r['precision']:.1%} | "
               f"{r['precision_at_monthly_fraud_rate']:.1%} |" for r in st) + "\n"
Path("RESULTS.md").write_text(md)
print(md)

fewer = 1 - h["alerts"] / r2["alerts"] if r2["alerts"] else 0
print("CV-ready figures (copy the real numbers, not these labels):")
print(f"  - Transactions analysed: {m['transactions_total']:,}")
print(f"  - Hybrid caught {h['recall']:.1%} of fraud at {h['precision']:.1%} alert precision "
      f"(false-positive rate {h['false_positive_rate']:.3%})")
print(f"  - Existing flag caught {base['recall']:.1%} of fraud")
print(f"  - {fewer:.0%} fewer alerts than the 2+ rules approach ({h['alerts']:,} vs {r2['alerts']:,})")
print(f"  - PR-AUC {pa['xgboost']:.3f}; {pa['xgboost_without_balance_fields']:.3f} without PaySim balance fields")
for r in st:
    print(f"  - Stress test, {r['features']}: recall {r['recall']:.1%}, precision {r['precision']:.1%}, "
          f"precision at monthly fraud rate {r['precision_at_monthly_fraud_rate']:.1%}, PR-AUC {r['pr_auc']:.3f}")
if "synthetic" in m["data_source"]:
    print("\n  WARNING: synthetic test data. Do not put these numbers on a CV.")
