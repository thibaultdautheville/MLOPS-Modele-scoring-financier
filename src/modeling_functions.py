import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate


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
