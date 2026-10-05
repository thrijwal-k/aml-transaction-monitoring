"""Alerts dashboard for the transaction monitoring pipeline. Run: streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUT = Path("outputs")
st.set_page_config(page_title="Transaction Monitoring Alerts", layout="wide")

if not (OUT / "metrics.json").exists():
    st.error("No results yet. Run `python run_pipeline.py --data data/paysim.csv` first.")
    st.stop()

m = json.loads((OUT / "metrics.json").read_text())
alerts = pd.read_csv(OUT / "alerts.csv")
rules = pd.read_csv(OUT / "rule_performance.csv")
daily = pd.read_csv(OUT / "daily_alerts.csv")
pr = pd.read_csv(OUT / "pr_curve.csv")
imp = pd.read_csv(OUT / "feature_importance.csv")

STRATEGY_NAMES = {
    "existing_paysim_flag": "Existing flag (PaySim baseline)",
    "rules_2plus": "Rules: 2+ scenarios",
    "rules_3plus": "Rules: 3+ scenarios",
    "xgboost": "Machine learning (XGBoost)",
    "hybrid_xgboost_or_3plus_rules": "Hybrid: ML or 3+ rules",
}

st.title("Transaction Monitoring Alerts")
st.caption(f"Data: {m['data_source']} · {m['transactions_total']:,} transactions · "
           f"evaluated on {m['test_transactions']:,} unseen TRANSFER/CASH_OUT transactions ({m['split']})")
if "synthetic" in m["data_source"]:
    st.warning("These results come from generated test data. Run the pipeline on the real PaySim file before "
               "quoting any figure.")

h = m["strategies"]["hybrid_xgboost_or_3plus_rules"]
base = m["strategies"]["existing_paysim_flag"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fraud in test period", f"{m['test_fraud']:,}")
c2.metric("Alerts raised", f"{h['alerts']:,}")
c3.metric("Fraud caught (recall)", f"{h['recall']:.1%}", f"{(h['recall'] - base['recall']) * 100:+.1f} pts vs existing flag")
c4.metric("Alert precision", f"{h['precision']:.1%}")
c5.metric("False-positive rate", f"{h['false_positive_rate']:.3%}")

tab_q, tab_perf, tab_rules, tab_model = st.tabs(["Alert queue", "Strategy comparison", "Rule scenarios", "Model"])

with tab_q:
    st.subheader("Investigator queue, highest risk first")
    f1, f2, f3 = st.columns([1, 2, 1])
    types = f1.multiselect("Type", sorted(alerts["type"].unique()), default=sorted(alerts["type"].unique()))
    codes = sorted({c.strip() for s in alerts["rules_fired"].fillna("") for c in s.split(",") if c.strip()})
    pick = f2.multiselect("Rules fired (any of)", codes)
    min_score = f3.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)
    q = alerts[alerts["type"].isin(types) & (alerts["risk_score"] >= min_score)]
    if pick:
        q = q[q["rules_fired"].fillna("").apply(lambda s: any(p in s for p in pick))]
    view = q.assign(outcome=q["isFraud"].map({1: "Confirmed fraud", 0: "False positive"}))[
        ["day", "type", "amount", "nameOrig", "nameDest", "risk_score", "rules_fired", "reasons", "outcome"]]
    st.dataframe(view.head(2000), width="stretch", hide_index=True,
                 column_config={"day": "Day", "type": "Type", "nameOrig": "Sender", "nameDest": "Receiver",
                                "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                                "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0,
                                                                              max_value=1, format="%.2f"),
                                "rules_fired": "Rules", "reasons": st.column_config.TextColumn("Top reasons",
                                                                                                width="large"),
                                "outcome": "Outcome"})
    st.caption(f"Showing {min(len(q), 2000):,} of {len(q):,} alerts. Reasons come from each alert's SHAP values.")

    fig = px.bar(daily, x="day", y=["alerts", "fraud"], barmode="group",
                 labels={"value": "Count", "day": "Day", "variable": ""}, title="Alerts and fraud by day")
    st.plotly_chart(fig, width="stretch")

with tab_perf:
    rows = [{"Strategy": STRATEGY_NAMES[k], "Alerts": v["alerts"], "Fraud caught": v["true_positives"],
             "Missed": v["missed_fraud"], "Recall": v["recall"], "Precision": v["precision"],
             "False-positive rate": v["false_positive_rate"]} for k, v in m["strategies"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent")
                                for c in ["Recall", "Precision", "False-positive rate"]})
    pa = m["pr_auc"]
    st.markdown(
        f"**Precision-recall AUC:** XGBoost {pa['xgboost']:.3f} · "
        f"XGBoost without balance fields {pa['xgboost_without_balance_fields']:.3f} · "
        f"Isolation Forest (no labels) {pa['isolation_forest']:.3f}")
    if "stress_test" in m:
        st.subheader("Stress test: what survives without simulator artefacts")
        sdf = pd.DataFrame(m["stress_test"])[["features", "pr_auc", "recall", "precision",
                                                "precision_at_monthly_fraud_rate", "alerts"]]
        st.dataframe(sdf, hide_index=True, width="stretch",
                     column_config={"features": "Features used", "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
                                    "recall": st.column_config.NumberColumn("Recall", format="percent"),
                                    "precision": st.column_config.NumberColumn("Precision (test days)", format="percent"),
                                    "precision_at_monthly_fraud_rate": st.column_config.NumberColumn(
                                        "Precision at monthly fraud rate", format="percent"),
                                    "alerts": "Alerts"})
        st.caption(f"Fraud is {m['fraud_rate_test_period']:.2%} of test-period transactions compared with "
                   f"{m['fraud_rate_monitored_month']:.2%} across the month, so test-period precision is optimistic.")
    st.info("Scepticism check: PaySim's balance fields are simulator artefacts that make fraud unusually easy to "
            "spot. The model without them shows how much of the performance survives on behaviour alone.")

with tab_rules:
    st.subheader("How each monitoring scenario performs on its own")
    st.dataframe(rules, hide_index=True, width="stretch",
                 column_config={"precision": st.column_config.NumberColumn(format="percent"),
                                "recall": st.column_config.NumberColumn(format="percent")})
    fig = px.scatter(rules, x="recall", y="precision", size="alerts", text="scenario", size_max=40,
                     title="Precision against recall by scenario (bubble size = alert volume)")
    fig.update_traces(textposition="top center")
    fig.update_xaxes(range=[-0.05, 1.1]); fig.update_yaxes(range=[-0.05, 1.1])
    st.plotly_chart(fig, width="stretch")

with tab_model:
    left, right = st.columns(2)
    fig = px.line(pr, x="recall", y="precision", title="Precision-recall curve (test period)")
    fig.update_xaxes(range=[0, 1.02]); fig.update_yaxes(range=[0, 1.02])
    left.plotly_chart(fig, width="stretch")
    fig = px.bar(imp.head(10).iloc[::-1], x="gain", y="label", orientation="h", title="What drives the risk score")
    right.plotly_chart(fig, width="stretch")
    st.caption(f"Alert threshold {m['threshold']:.3f}, chosen to maximise F2 on the last four training days "
               "(recall weighted above precision, since a missed fraud costs more than an extra review).")
