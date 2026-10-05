"""Transaction monitoring pipeline: rules + machine learning + explainable alerts.

Usage:
    python run_pipeline.py --data data/paysim.csv      # real PaySim data (report these results)
    python run_pipeline.py --synthetic                 # quick test on generated data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve, roc_auc_score

from src.data import load_paysim, make_synthetic
from src.features import FEATURES, REASON_LABELS, add_features
from src.rules import SCENARIOS, apply_rules, rule_performance

OUT = Path("outputs")
TRAIN_END_STEP = 480  # days 1-20 train, days 21-31 test (time-based split, no look-ahead)
VAL_START_STEP = 384  # last 4 training days are used to pick the alert threshold
BALANCE_FEATURES = ["orig_balance_error", "dest_balance_error", "dest_zero_balances", "orig_zero_balance",
                    "account_emptied", "amount_to_balance"]


def binary_metrics(y: np.ndarray, flag: np.ndarray) -> dict:
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    return {
        "alerts": tp + fp, "true_positives": tp, "false_positives": fp, "missed_fraud": fn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def train_xgb(X_tr, y_tr, X_val, y_val, seed=42):
    pos = max(int(y_tr.sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, scale_pos_weight=(len(y_tr) - pos) / pos, tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=40, random_state=seed, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def pick_threshold(y_val, p_val) -> float:
    """Choose the score cut-off that maximises F2 on validation data (recall weighted twice precision,
    because a missed fraud costs more than an extra alert for an investigator)."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-12)
    return float(thr[np.argmax(f2[:-1])])


def explain(model: xgb.XGBClassifier, X: pd.DataFrame, top: int = 3) -> list[str]:
    """Top reasons per alert from XGBoost's built-in SHAP contributions."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    names = np.array([REASON_LABELS[c] for c in X.columns])
    order = np.argsort(-contrib, axis=1)[:, :top]
    return ["; ".join(names[i][contrib[r, i] > 0]) for r, i in enumerate(order)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paysim.csv")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    raw = make_synthetic() if args.synthetic else load_paysim(args.data)
    source = "synthetic test data (do not report)" if args.synthetic else "PaySim"
    print(f"Loaded {len(raw):,} transactions, {int(raw['isFraud'].sum()):,} fraud ({source})")

    d = add_features(raw)
    r = apply_rules(d)
    d = pd.concat([d, r], axis=1)
    y = d["isFraud"].to_numpy()
    print(f"Monitoring {len(d):,} TRANSFER/CASH_OUT transactions, features built in {time.time() - t0:.0f}s")

    train = d["step"] < TRAIN_END_STEP
    val = train & (d["step"] >= VAL_START_STEP)
    fit = train & ~val
    test = ~train
    X = d[FEATURES]

    # 1) Supervised model
    model = train_xgb(X[fit], y[fit], X[val], y[val])
    thr = pick_threshold(y[val], model.predict_proba(X[val])[:, 1])
    p_test = model.predict_proba(X[test])[:, 1]
    yt = y[test]
    model_flag = p_test >= thr

    # 2) Scepticism check: how much does the model lean on PaySim's balance fields?
    robust_cols = [c for c in FEATURES if c not in BALANCE_FEATURES]
    robust = train_xgb(X.loc[fit, robust_cols], y[fit], X.loc[val, robust_cols], y[val])
    p_robust = robust.predict_proba(X.loc[test, robust_cols])[:, 1]

    # 2b) Stress test: remove simulator artefacts one group at a time and see what survives
    prior = float(y.mean())  # fraud rate across the whole month of monitored transactions
    stress_sets = {
        "All features": FEATURES,
        "Without balance fields": robust_cols,
        "Without balance fields or pass-through pairing": [c for c in robust_cols if c != "matched_pass_through"],
        "Behaviour only (amount, type, receiver activity)": ["is_transfer", "log_amount", "dest_fan_in_day",
                                                              "dest_tx_day", "dest_amount_day_log"],
    }
    stress = []
    for name, cols in stress_sets.items():
        mdl = model if cols == FEATURES else train_xgb(X.loc[fit, cols], y[fit], X.loc[val, cols], y[val])
        th = thr if cols == FEATURES else pick_threshold(y[val], mdl.predict_proba(X.loc[val, cols])[:, 1])
        ps = mdl.predict_proba(X.loc[test, cols])[:, 1]
        bm = binary_metrics(yt, ps >= th)
        # Precision if fraud were as rare as across the whole month, not the fraud-heavy final days
        tpr, fpr = bm["recall"], bm["false_positive_rate"]
        adj = tpr * prior / (tpr * prior + fpr * (1 - prior)) if tpr or fpr else 0.0
        stress.append({"features": name, "n_features": len(cols), "pr_auc": float(average_precision_score(yt, ps)),
                       **bm, "precision_at_monthly_fraud_rate": adj})
        print(f"stress: {name:50s} PR-AUC={stress[-1]['pr_auc']:.3f} recall={bm['recall']:.3f} "
              f"precision={bm['precision']:.3f} adj_precision={adj:.3f}")

    # 3) Unsupervised anomaly detection (no labels), for comparison
    iso = IsolationForest(n_estimators=200, max_samples=50_000, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X[fit & (d["isFraud"] == 0)])
    p_iso = -iso.score_samples(X[test])

    # 4) Rules, and the hybrid alerting strategy
    rules_test = d.loc[test, list(SCENARIOS) + ["rule_hits"]]
    rule_flag = (rules_test["rule_hits"] >= 2).to_numpy()
    rule3_flag = (rules_test["rule_hits"] >= 3).to_numpy()
    hybrid_flag = model_flag | rule3_flag  # strong multi-rule hits are always reviewed, even if the model misses them
    paysim_flag = (d.loc[test, "isFlaggedFraud"] == 1).to_numpy()

    metrics = {
        "data_source": source,
        "transactions_total": int(len(raw)),
        "transactions_monitored": int(len(d)),
        "fraud_total": int(raw["isFraud"].sum()),
        "test_transactions": int(test.sum()),
        "test_fraud": int(yt.sum()),
        "fraud_rate_monitored_month": float(y.mean()),
        "fraud_rate_test_period": float(yt.mean()),
        "stress_test": stress,
        "split": f"train steps < {TRAIN_END_STEP} (threshold tuned on steps {VAL_START_STEP}-{TRAIN_END_STEP - 1}), "
                 f"test steps >= {TRAIN_END_STEP}",
        "threshold": thr,
        "pr_auc": {
            "xgboost": float(average_precision_score(yt, p_test)),
            "xgboost_without_balance_fields": float(average_precision_score(yt, p_robust)),
            "isolation_forest": float(average_precision_score(yt, p_iso)),
        },
        "roc_auc_xgboost": float(roc_auc_score(yt, p_test)),
        "f2_xgboost": float(fbeta_score(yt, model_flag, beta=2)),
        "strategies": {
            "existing_paysim_flag": binary_metrics(yt, paysim_flag),
            "rules_2plus": binary_metrics(yt, rule_flag),
            "rules_3plus": binary_metrics(yt, rule3_flag),
            "xgboost": binary_metrics(yt, model_flag),
            "hybrid_xgboost_or_3plus_rules": binary_metrics(yt, hybrid_flag),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }

    # Alerts with reasons, ranked by risk score for the investigator queue
    td = d.loc[test].copy()
    td["risk_score"] = p_test
    td["alert"] = hybrid_flag
    alerts = td[td["alert"]].sort_values("risk_score", ascending=False)
    alerts["reasons"] = explain(model, alerts[FEATURES]) if len(alerts) else []
    cols = ["step", "day", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig",
            "risk_score", "rules_fired", "rule_hits", "reasons", "isFraud"]
    alerts[cols].to_csv(OUT / "alerts.csv", index=False)

    rule_performance(d.loc[test, list(SCENARIOS) + ["rule_hits"]], d.loc[test, "isFraud"]).to_csv(
        OUT / "rule_performance.csv", index=False)
    prec, rec, thr_curve = precision_recall_curve(yt, p_test)
    idx = np.linspace(0, len(thr_curve) - 1, min(400, len(thr_curve))).astype(int)
    pd.DataFrame({"threshold": thr_curve[idx], "precision": prec[idx], "recall": rec[idx]}).to_csv(
        OUT / "pr_curve.csv", index=False)
    td.groupby("day").agg(transactions=("amount", "size"), alerts=("alert", "sum"),
                          fraud=("isFraud", "sum")).reset_index().to_csv(OUT / "daily_alerts.csv", index=False)
    imp = model.get_booster().get_score(importance_type="gain")
    pd.DataFrame({"feature": list(imp), "label": [REASON_LABELS[f] for f in imp], "gain": list(imp.values())}) \
        .sort_values("gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    s = metrics["strategies"]
    print(json.dumps(metrics["pr_auc"], indent=2))
    for k, v in s.items():
        print(f"{k:28s} alerts={v['alerts']:>8,}  recall={v['recall']:.3f}  precision={v['precision']:.3f}  "
              f"FPR={v['false_positive_rate']:.5f}")
    print(f"Done in {metrics['runtime_seconds']}s. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
