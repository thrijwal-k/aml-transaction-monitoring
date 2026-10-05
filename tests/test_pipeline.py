"""Basic checks that rules and features behave as described. Run: python -m pytest -q"""
import pandas as pd

from src.data import make_synthetic
from src.features import FEATURES, add_features
from src.rules import SCENARIOS, apply_rules


def _tx(**kw):
    base = dict(step=1, type="TRANSFER", amount=1000.0, nameOrig="C1", oldbalanceOrg=5000.0,
                newbalanceOrig=4000.0, nameDest="C2", oldbalanceDest=0.0, newbalanceDest=1000.0,
                isFraud=0, isFlaggedFraud=0)
    base.update(kw)
    return base


def test_account_emptied_rule():
    df = pd.DataFrame([_tx(amount=5000.0, newbalanceOrig=0.0, newbalanceDest=5000.0)])
    r = apply_rules(add_features(df))
    assert r["R2_ACCOUNT_EMPTIED"].iloc[0] == 1


def test_pass_through_matches_transfer_and_cash_out():
    df = pd.DataFrame([_tx(amount=777.77), _tx(type="CASH_OUT", step=2, amount=777.77, nameOrig="C3")])
    d = add_features(df)
    assert d["matched_pass_through"].tolist() == [1, 1]


def test_payments_are_out_of_scope():
    df = pd.DataFrame([_tx(type="PAYMENT", nameDest="M1")])
    assert len(add_features(df)) == 0


def test_synthetic_pipeline_shapes():
    d = add_features(make_synthetic(20_000))
    r = apply_rules(d)
    assert set(FEATURES) <= set(d.columns)
    assert set(SCENARIOS) <= set(r.columns)
    assert d["isFraud"].sum() > 0
