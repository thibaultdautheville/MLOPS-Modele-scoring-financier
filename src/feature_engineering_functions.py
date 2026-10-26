"""Feature engineering Home Credit.

Adapté du kernel Kaggle "LightGBM with Simple Features" (jsaguiar) :
https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script

Principales adaptations par rapport au kernel original :
- pandas >= 2 : `DataFrame.append` n'existe plus -> `pd.concat` ;
- suppression de la colonne parasite `index` créée par `reset_index()` dans le kernel ;
- encodage binaire explicite (F=0/M=1, N=0/Y=1) au lieu de `pd.factorize`, dont le
  sens dépend de l'ordre d'apparition (illisible pour l'explicabilité) ;
- indicateur DAYS_EMPLOYED_ANOM (idée du kernel "Start Here: A Gentle Introduction") ;
- DPD/DBD calculés avec `clip` (vectorisé) au lieu d'un `apply` ligne à ligne ;
- réduction mémoire float32 et libération des tables intermédiaires (8 Go de RAM).
"""

import gc

import numpy as np
import pandas as pd

from src.data_processing_functions import load_csv, one_hot_encoder, reduce_memory_usage
from src.settings import (
    APPLICATION_TEST_FILE,
    APPLICATION_TRAIN_FILE,
    BUREAU_BALANCE_FILE,
    BUREAU_FILE,
    CREDIT_CARD_FILE,
    ID_COLUMN,
    INSTALLMENTS_FILE,
    POS_CASH_FILE,
    PREVIOUS_APPLICATION_FILE,
)

# Valeur sentinelle utilisée par Home Credit pour "date non renseignée" (~1000 ans)
DAYS_ANOMALY_VALUE = 365243


def _flatten_columns(agg: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """('AMT_CREDIT', 'mean') -> 'PREFIX_AMT_CREDIT_MEAN'."""
    agg.columns = pd.Index([f"{prefix}{col}_{stat.upper()}" for col, stat in agg.columns])
    return agg


# ---------------------- Table principale : demandes de crédit ---------------------
def application_train_test(num_rows: int | None = None, nan_as_category: bool = False):
    train = load_csv(APPLICATION_TRAIN_FILE, num_rows)
    test = load_csv(APPLICATION_TEST_FILE, num_rows)
    # Train et test sont empilés pour appliquer exactement le même encodage.
    # Aucune fuite de la cible : les transformations ci-dessous sont ligne à ligne.
    df = pd.concat([train, test], ignore_index=True)
    del train, test

    n_before = len(df)
    df = df[df["CODE_GENDER"] != "XNA"].copy()
    print(f"Lignes supprimées (CODE_GENDER = XNA) : {n_before - len(df)}")

    df["CODE_GENDER"] = df["CODE_GENDER"].map({"F": 0, "M": 1}).astype(np.uint8)
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col] = df[col].map({"N": 0, "Y": 1}).astype(np.uint8)

    df, _ = one_hot_encoder(df, nan_as_category)

    # DAYS_EMPLOYED = 365243 : retraités / sans emploi (≈ 18 % des lignes)
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_ANOMALY_VALUE).astype(np.uint8)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_ANOMALY_VALUE, np.nan)

    # Ratios métier du kernel
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    return reduce_memory_usage(df)


# ---------------------- Bureau de crédit (autres institutions) ---------------------
def bureau_and_balance(num_rows: int | None = None, nan_as_category: bool = True):
    bureau = load_csv(BUREAU_FILE, num_rows)
    bb = load_csv(BUREAU_BALANCE_FILE, num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Historique mensuel -> une ligne par crédit externe
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_aggregations[col] = ["mean"]
    bb_agg = _flatten_columns(bb.groupby("SK_ID_BUREAU").agg(bb_aggregations), "")
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau = bureau.drop(columns=["SK_ID_BUREAU"])
    del bb, bb_agg
    gc.collect()

    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    cat_aggregations = {cat: ["mean"] for cat in bureau_cat}
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ["mean"]

    # Crédit externe -> une ligne par client, puis zoom sur crédits actifs / clôturés
    bureau_agg = _flatten_columns(
        bureau.groupby(ID_COLUMN).agg({**num_aggregations, **cat_aggregations}), "BURO_"
    )
    active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    active_agg = _flatten_columns(active.groupby(ID_COLUMN).agg(num_aggregations), "ACTIVE_")
    bureau_agg = bureau_agg.join(active_agg, how="left")
    closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
    closed_agg = _flatten_columns(closed.groupby(ID_COLUMN).agg(num_aggregations), "CLOSED_")
    bureau_agg = bureau_agg.join(closed_agg, how="left")
    del bureau, active, active_agg, closed, closed_agg
    gc.collect()
    return bureau_agg


# ---------------------- Demandes précédentes chez Home Credit ---------------------
def previous_applications(num_rows: int | None = None, nan_as_category: bool = True):
    prev = load_csv(PREVIOUS_APPLICATION_FILE, num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category)

    for col in [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]:
        prev[col] = prev[col].replace(DAYS_ANOMALY_VALUE, np.nan)
    # Montant demandé / montant finalement accordé
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    num_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    cat_aggregations = {cat: ["mean"] for cat in cat_cols}

    prev_agg = _flatten_columns(
        prev.groupby(ID_COLUMN).agg({**num_aggregations, **cat_aggregations}), "PREV_"
    )
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    approved_agg = _flatten_columns(
        approved.groupby(ID_COLUMN).agg(num_aggregations), "APPROVED_"
    )
    prev_agg = prev_agg.join(approved_agg, how="left")
    refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
    refused_agg = _flatten_columns(refused.groupby(ID_COLUMN).agg(num_aggregations), "REFUSED_")
    prev_agg = prev_agg.join(refused_agg, how="left")
    del prev, approved, approved_agg, refused, refused_agg
    gc.collect()
    return prev_agg


# ---------------------- Soldes mensuels POS / prêts cash ---------------------
def pos_cash(num_rows: int | None = None, nan_as_category: bool = True):
    pos = load_csv(POS_CASH_FILE, num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)
    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    pos_agg = _flatten_columns(pos.groupby(ID_COLUMN).agg(aggregations), "POS_")
    pos_agg["POS_COUNT"] = pos.groupby(ID_COLUMN).size()
    del pos
    gc.collect()
    return pos_agg


# ---------------------- Historique des remboursements ---------------------
def installments_payments(num_rows: int | None = None, nan_as_category: bool = True):
    ins = load_csv(INSTALLMENTS_FILE, num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)

    # Part payée et écart au montant dû pour chaque échéance
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    # Jours de retard (DPD) et jours d'avance (DBD), bornés à 0
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    ins_agg = _flatten_columns(ins.groupby(ID_COLUMN).agg(aggregations), "INSTAL_")
    ins_agg["INSTAL_COUNT"] = ins.groupby(ID_COLUMN).size()
    del ins
    gc.collect()
    return ins_agg


# ---------------------- Soldes mensuels cartes de crédit ---------------------
def credit_card_balance(num_rows: int | None = None, nan_as_category: bool = True):
    cc = load_csv(CREDIT_CARD_FILE, num_rows)
    cc, _ = one_hot_encoder(cc, nan_as_category)
    cc = cc.drop(columns=["SK_ID_PREV"])
    cc_agg = _flatten_columns(
        cc.groupby(ID_COLUMN).agg(["min", "max", "mean", "sum", "var"]), "CC_"
    )
    cc_agg["CC_COUNT"] = cc.groupby(ID_COLUMN).size()
    del cc
    gc.collect()
    return cc_agg


# ---------------------- Pipeline complet ---------------------
def build_features(num_rows: int | None = None) -> pd.DataFrame:
    """Enchaîne les 6 blocs et les joint sur SK_ID_CURR (jointures à gauche).

    Une jointure à gauche garantit qu'aucun client de la table principale n'est
    perdu : un client sans historique externe reçoit simplement des NaN.
    """
    df = application_train_test(num_rows)
    n_rows = len(df)
    print(f"Table principale : {df.shape}")

    blocks = [
        ("bureau", bureau_and_balance),
        ("previous_applications", previous_applications),
        ("pos_cash", pos_cash),
        ("installments", installments_payments),
        ("credit_card", credit_card_balance),
    ]
    for name, function in blocks:
        agg = reduce_memory_usage(function(num_rows))
        coverage = df[ID_COLUMN].isin(agg.index).mean()
        df = df.join(agg, how="left", on=ID_COLUMN)
        assert len(df) == n_rows, f"La jointure {name} a modifié le nombre de lignes"
        print(f"+ {name:<22} {agg.shape[1]:>4} features | clients couverts : {coverage:.1%}")
        del agg
        gc.collect()

    print(f"Dataset final : {df.shape}")
    # Les compteurs entiers passent en float64 quand la jointure à gauche crée des NaN
    return reduce_memory_usage(df)
