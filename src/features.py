"""Transaction-level features for monitoring. All vectorised so they run on the full 6.36m-row PaySim file."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONITORED_TYPES = ["TRANSFER", "CASH_OUT"]  # every fraud case in PaySim is one of these two types

FEATURES = [
    "is_transfer", "log_amount", "hour", "amount_to_balance", "account_emptied",
    "orig_balance_error", "dest_balance_error", "dest_zero_balances", "orig_zero_balance",
    "matched_pass_through", "dest_fan_in_day", "dest_tx_day", "dest_amount_day_log",
]

# Plain-English labels used as alert reasons in the dashboard
REASON_LABELS = {
    "is_transfer": "Transfer between customer accounts",
    "log_amount": "Unusually large amount",
    "hour": "Time of day",
    "amount_to_balance": "Amount large relative to sender balance",
    "account_emptied": "Sender account emptied",
    "orig_balance_error": "Sender balance does not reconcile",
    "dest_balance_error": "Receiver balance does not reconcile",
    "dest_zero_balances": "Receiver shows zero balance before and after",
    "orig_zero_balance": "Sender had no recorded balance",
    "matched_pass_through": "Same amount cashed out straight after transfer",
    "dest_fan_in_day": "Receiver getting money from many senders today",
    "dest_tx_day": "High activity on receiver account today",
    "dest_amount_day_log": "Large total received by receiver today",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return monitored transactions (TRANSFER and CASH_OUT) with model features added."""
    d = df[df["type"].isin(MONITORED_TYPES)].copy()
    d["type"] = d["type"].astype(str)
    amt = d["amount"].to_numpy()

    d["is_transfer"] = (d["type"] == "TRANSFER").astype("int8")
    d["log_amount"] = np.log1p(amt)
    d["hour"] = (d["step"] % 24).astype("int8")
    d["day"] = (d["step"] // 24).astype("int16")
    d["amount_to_balance"] = amt / (d["oldbalanceOrg"].to_numpy() + 1.0)
    d["account_emptied"] = ((d["oldbalanceOrg"] > 0) & (d["newbalanceOrig"] <= 0.01)
                            & (amt >= 0.99 * d["oldbalanceOrg"])).astype("int8")
    # Reconciliation checks: does each balance move by the amount sent or received?
    d["orig_balance_error"] = d["newbalanceOrig"] + amt - d["oldbalanceOrg"]
    d["dest_balance_error"] = d["oldbalanceDest"] + amt - d["newbalanceDest"]
    d["dest_zero_balances"] = ((d["oldbalanceDest"] == 0) & (d["newbalanceDest"] == 0)).astype("int8")
    d["orig_zero_balance"] = (d["oldbalanceOrg"] == 0).astype("int8")

    # Rapid movement of funds: a TRANSFER and a CASH_OUT of exactly the same amount within one hour (step)
    key = d["amount"].round(2)
    tr = pd.DataFrame({"k": key[d["type"] == "TRANSFER"], "s": d.loc[d["type"] == "TRANSFER", "step"]})
    co = pd.DataFrame({"k": key[d["type"] == "CASH_OUT"], "s": d.loc[d["type"] == "CASH_OUT", "step"]})
    tr_keys = set(zip(tr["k"], tr["s"], strict=True)) | set(zip(tr["k"], tr["s"] + 1, strict=True))
    co_keys = set(zip(co["k"], co["s"], strict=True)) | set(zip(co["k"], co["s"] - 1, strict=True))
    is_tr = (d["type"] == "TRANSFER").to_numpy()
    pairs = list(zip(key.to_numpy(), d["step"].to_numpy(), strict=True))
    d["matched_pass_through"] = np.fromiter(
        ((p in co_keys) if t else (p in tr_keys) for p, t in zip(pairs, is_tr, strict=True)), dtype=bool, count=len(d)
    ).astype("int8")

    # Receiver behaviour within the same day (a daily monitoring batch)
    g = d.groupby(["nameDest", "day"], observed=True)
    d["dest_fan_in_day"] = g["nameOrig"].transform("nunique").astype("int32")
    d["dest_tx_day"] = g["amount"].transform("size").astype("int32")
    d["dest_amount_day_log"] = np.log1p(g["amount"].transform("sum"))
    return d.reset_index(drop=True)
