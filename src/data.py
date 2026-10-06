"""Load PaySim transactions, or generate a small synthetic sample with the same schema for testing.

PaySim (Lopez-Rojas et al., 2016) simulates one month of mobile money transactions.
Download it free from Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1
and save the CSV as data/paysim.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

DTYPES = {
    "step": "int32", "type": "category", "amount": "float64",
    "nameOrig": "string", "oldbalanceOrg": "float64", "newbalanceOrig": "float64",
    "nameDest": "string", "oldbalanceDest": "float64", "newbalanceDest": "float64",
    "isFraud": "int8", "isFlaggedFraud": "int8",
}


def load_paysim(path: str | Path) -> pd.DataFrame:
    """Read the PaySim CSV with compact dtypes (about 1 GB in memory for all 6.36m rows)."""
    df = pd.read_csv(path, dtype=DTYPES)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Not a PaySim file, missing columns: {sorted(missing)}")
    return df.sort_values("step", kind="stable").reset_index(drop=True)


def make_synthetic(n_tx: int = 300_000, fraud_rate: float = 0.0013, seed: int = 7) -> pd.DataFrame:
    """Generate PaySim-shaped data for testing the pipeline. Results on it mean nothing:
    use the real PaySim file for any figure you report."""
    rng = np.random.default_rng(seed)
    n_cust, n_merch = n_tx // 2, n_tx // 20
    types = np.array(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    p_types = np.array([0.22, 0.352, 0.0065, 0.338, 0.0835])
    n_legit = int(n_tx * (1 - 2 * fraud_rate))

    t = rng.choice(types, size=n_legit, p=p_types / p_types.sum())
    step = rng.integers(1, 744, size=n_legit)
    amount = np.round(rng.lognormal(mean=11.0, sigma=1.4, size=n_legit), 2)
    amount[t == "PAYMENT"] = np.round(rng.lognormal(9.0, 1.0, (t == "PAYMENT").sum()), 2)
    old_o = np.round(rng.lognormal(10.5, 2.0, n_legit) * (rng.random(n_legit) > 0.3), 2)
    out = np.isin(t, ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    new_o = np.where(out, np.maximum(old_o - amount, 0), old_o + amount)
    # PaySim quirk: many legitimate rows also carry zero balances
    new_o[rng.random(n_legit) < 0.05] = 0
    orig = np.char.add("C", rng.integers(1e8, 1e8 + n_cust, n_legit).astype(str))
    is_merch = t == "PAYMENT"
    dest = np.where(is_merch,
                    np.char.add("M", rng.integers(1e8, 1e8 + n_merch, n_legit).astype(str)),
                    np.char.add("C", rng.integers(1e8, 1e8 + n_cust, n_legit).astype(str)))
    old_d = np.where(is_merch, 0, np.round(rng.lognormal(11.5, 2.0, n_legit), 2))
    new_d = np.where(is_merch, 0, old_d + np.where(np.isin(t, ["CASH_OUT", "TRANSFER"]), amount, -amount).clip(min=-old_d))
    new_d[(~is_merch) & (rng.random(n_legit) < 0.08)] = 0  # missing balance updates happen in legit data too

    # Legitimate behaviour that looks suspicious: people often move money and withdraw it straight away
    n_pairs = int(n_legit * 0.004)
    i_tr = rng.choice(np.where(t == "TRANSFER")[0], n_pairs, replace=False)
    i_co = rng.choice(np.where(t == "CASH_OUT")[0], n_pairs, replace=False)
    amount[i_co] = amount[i_tr]
    step[i_co] = step[i_tr]
    new_o[i_co] = np.maximum(old_o[i_co] - amount[i_co], 0)
    legit = pd.DataFrame({
        "step": step, "type": t, "amount": amount, "nameOrig": orig,
        "oldbalanceOrg": old_o, "newbalanceOrig": new_o, "nameDest": dest,
        "oldbalanceDest": old_d, "newbalanceDest": new_d, "isFraud": 0,
    })

    # Fraud: take over a customer account, empty it with a TRANSFER to a mule, then CASH_OUT the same amount
    n_fraud = int(n_tx * fraud_rate)
    f_step = rng.integers(1, 744, size=n_fraud)
    balance = np.round(rng.lognormal(12.0, 1.3, n_fraud), 2)
    partial = rng.random(n_fraud) < 0.25  # some fraud does not empty the account
    f_amt = np.where(partial, np.round(balance * rng.uniform(0.3, 0.9, n_fraud), 2), balance)
    f_amt = np.minimum(f_amt, 10_000_000)
    victims = np.char.add("C", rng.integers(2e8, 2e8 + n_fraud * 3, n_fraud).astype(str))
    mules = np.char.add("C", rng.integers(3e8, 3e8 + max(n_fraud // 4, 1), n_fraud).astype(str))
    cashers = np.char.add("C", rng.integers(4e8, 4e8 + n_fraud * 3, n_fraud).astype(str))
    mule_old = np.where(rng.random(n_fraud) < 0.6, 0, np.round(rng.lognormal(10, 2, n_fraud), 2))
    mule_new = np.where(rng.random(n_fraud) < 0.6, 0, mule_old + f_amt)
    transfer = pd.DataFrame({
        "step": f_step, "type": "TRANSFER", "amount": f_amt, "nameOrig": victims,
        "oldbalanceOrg": balance, "newbalanceOrig": np.round(balance - f_amt, 2), "nameDest": mules,
        "oldbalanceDest": mule_old, "newbalanceDest": mule_new, "isFraud": 1,
    })
    cash_old_d = np.round(rng.lognormal(11, 2, n_fraud), 2)
    # Not every fraud is cashed out the same way: some amounts are split or delayed
    co_amt = np.where(rng.random(n_fraud) < 0.3, np.round(f_amt * rng.uniform(0.5, 0.95, n_fraud), 2), f_amt)
    cashout = pd.DataFrame({
        "step": f_step + rng.choice([0, 1, 3, 8], n_fraud, p=[0.5, 0.25, 0.15, 0.1]), "type": "CASH_OUT",
        "amount": co_amt,
        "nameOrig": cashers, "oldbalanceOrg": balance, "newbalanceOrig": np.round(balance - co_amt, 2),
        "nameDest": np.char.add("C", rng.integers(5e8, 5e8 + n_fraud, n_fraud).astype(str)),
        "oldbalanceDest": cash_old_d, "newbalanceDest": cash_old_d + co_amt, "isFraud": 1,
    })
    df = pd.concat([legit, transfer, cashout], ignore_index=True)
    df["isFlaggedFraud"] = ((df["type"] == "TRANSFER") & (df["amount"] > 200_000) & (df["isFraud"] == 1)
                            & (rng.random(len(df)) < 0.002)).astype("int8")
    df["type"] = df["type"].astype("category")
    for c in ["nameOrig", "nameDest"]:
        df[c] = df[c].astype("string")
    df["step"] = np.minimum(df["step"], 743).astype("int32")
    df["isFraud"] = df["isFraud"].astype("int8")
    return df[COLUMNS].sort_values("step", kind="stable").reset_index(drop=True)
