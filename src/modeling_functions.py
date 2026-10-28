import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

from src.settings import COST_FN, COST_FP

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ---------------------- Validation croisée ---------------------
def perform_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    cross_val_type,
    scoring_metrics: tuple,
    groups=None,
):
    """Validation croisée puis réentraînement sur tout le jeu (screencast p3_c4_mlflow.py).

    Renvoie les scores bruts, un dictionnaire moyenne / écart-type par métrique (prêt pour
    mlflow.log_metrics) et le modèle réentraîné.
    """
    scores = cross_validate(
        model,
        X,
        y,
        cv=cross_val_type,
        return_train_score=True,
        return_estimator=True,
        scoring=scoring_metrics,
        groups=groups,
    )

    scores_dict = {}
    for metric in scoring_metrics:
        scores_dict["average_train_" + metric] = np.mean(scores["train_" + metric])
        scores_dict["train_" + metric + "_std"] = np.std(scores["train_" + metric])
        scores_dict["average_test_" + metric] = np.mean(scores["test_" + metric])
        scores_dict["test_" + metric + "_std"] = np.std(scores["test_" + metric])

    model.fit(X, y)

    return scores, scores_dict, model


# ---------------------- Coût métier et seuil de décision ---------------------
def business_cost(y_true, y_pred, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Coût métier moyen par client : (cost_fn x FN + cost_fp x FP) / n.

    Exprimé en « unités de faux positif » : 1 = coût d'un bon client refusé.
    """
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (cost_fn * fn + cost_fp * fp) / len(y_true)


def cost_by_threshold(y_true, y_proba, thresholds=THRESHOLDS) -> pd.Series:
    """Coût métier pour chaque seuil : un client est classé en défaut si proba >= seuil."""
    return pd.Series(
        [business_cost(y_true, (y_proba >= t).astype(int)) for t in thresholds],
        index=pd.Index(thresholds, name="seuil"),
        name="cout_metier",
    )


def find_best_threshold(y_true, y_proba, thresholds=THRESHOLDS) -> tuple[float, float]:
    """Seuil qui minimise le coût métier, et le coût correspondant."""
    costs = cost_by_threshold(y_true, y_proba, thresholds)
    return float(costs.idxmin()), float(costs.min())


def classification_metrics(y_true, y_pred) -> dict:
    """Métriques de classification à un seuil donné (classe positive = défaut)."""
    return {
        "business_cost": business_cost(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


# ---------------------- Validation croisée avec prédictions hors pli ---------------------
def cross_validate_oof(model, X: pd.DataFrame, y: pd.Series, cv) -> tuple[np.ndarray, list, dict]:
    """Validation croisée qui conserve les probabilités hors pli (out-of-fold).

    Chaque client du train reçoit une probabilité prédite par un modèle qui ne l'a pas vu :
    on peut ainsi choisir le seuil de décision sans fuite de données, puis mesurer le coût
    métier pli par pli. Renvoie les probabilités hors pli, les indices de validation de
    chaque pli et l'AUC train / validation (pour détecter le surapprentissage).
    """
    oof_proba = np.zeros(len(y))
    val_indices = []
    fold_scores = {"train_roc_auc": [], "test_roc_auc": [], "fit_time": []}

    for train_idx, val_idx in cv.split(X, y):
        fold_model = clone(model)
        start = time.time()
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_scores["fit_time"].append(time.time() - start)

        train_proba = fold_model.predict_proba(X.iloc[train_idx])[:, 1]
        oof_proba[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_scores["train_roc_auc"].append(roc_auc_score(y.iloc[train_idx], train_proba))
        fold_scores["test_roc_auc"].append(roc_auc_score(y.iloc[val_idx], oof_proba[val_idx]))
        val_indices.append(val_idx)

    return oof_proba, val_indices, fold_scores


def scores_at_threshold(y: pd.Series, oof_proba, val_indices: list, threshold: float) -> dict:
    """Moyenne et écart-type par pli des métriques au seuil choisi (robustesse du modèle)."""
    per_fold = pd.DataFrame(
        [
            classification_metrics(y.iloc[idx], (oof_proba[idx] >= threshold).astype(int))
            for idx in val_indices
        ]
    )
    scores = {}
    for metric in per_fold.columns:
        scores["average_test_" + metric] = per_fold[metric].mean()
        scores["test_" + metric + "_std"] = per_fold[metric].std(ddof=0)
    return scores
