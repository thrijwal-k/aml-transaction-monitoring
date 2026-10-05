"""Rule-based transaction monitoring scenarios, the way a bank's first line of detection works."""
from __future__ import annotations

import pandas as pd

LARGE_TRANSFER_THRESHOLD = 200_000  # same threshold as PaySim's built-in flag
FAN_IN_THRESHOLD = 3  # distinct senders to one receiver in a day

SCENARIOS = {
    "R1_LARGE_TRANSFER": "Transfer above 200,000",
    "R2_ACCOUNT_EMPTIED": "Sender account emptied in one transaction",
    "R3_RAPID_PASS_THROUGH": "Same amount transferred and cashed out within one hour",
    "R4_BALANCE_MISMATCH": "Receiver balance does not move by the amount received",
    "R5_FAN_IN": f"Receiver gets money from {FAN_IN_THRESHOLD}+ senders in one day",
}


def apply_rules(d: pd.DataFrame) -> pd.DataFrame:
    """Add one 0/1 column per scenario plus a hit count and a readable list of the rules that fired."""
    out = pd.DataFrame(index=d.index)
    out["R1_LARGE_TRANSFER"] = (d["is_transfer"] == 1) & (d["amount"] > LARGE_TRANSFER_THRESHOLD)
    out["R2_ACCOUNT_EMPTIED"] = d["account_emptied"] == 1
    out["R3_RAPID_PASS_THROUGH"] = d["matched_pass_through"] == 1
    out["R4_BALANCE_MISMATCH"] = (d["dest_balance_error"].abs() > 0.01 * d["amount"]) & (d["is_transfer"] == 1)
    out["R5_FAN_IN"] = d["dest_fan_in_day"] >= FAN_IN_THRESHOLD
    out = out.astype("int8")
    out["rule_hits"] = out[list(SCENARIOS)].sum(axis=1).astype("int8")
    names = out[list(SCENARIOS)].astype(bool)
    fired = pd.Series("", index=out.index)
    for c in SCENARIOS:
        fired = fired.where(~names[c], fired + ", " + c.split("_", 1)[0])
    out["rules_fired"] = fired.str.lstrip(", ")
    return out


def rule_performance(rules: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Alerts, true positives, precision and recall for each scenario and for common combinations."""
    rows = []
    total_fraud = int(y.sum())
    checks = {c: rules[c] == 1 for c in SCENARIOS}
    checks["Any 1+ rule"] = rules["rule_hits"] >= 1
    checks["Any 2+ rules"] = rules["rule_hits"] >= 2
    for name, mask in checks.items():
        alerts = int(mask.sum())
        tp = int((mask & (y == 1)).sum())
        rows.append({
            "scenario": name,
            "description": SCENARIOS.get(name, "Combined rule logic"),
            "alerts": alerts,
            "true_positives": tp,
            "precision": tp / alerts if alerts else 0.0,
            "recall": tp / total_fraud if total_fraud else 0.0,
            "false_positives": alerts - tp,
        })
    return pd.DataFrame(rows)
