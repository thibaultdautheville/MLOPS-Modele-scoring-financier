import re

import numpy as np
import pandas as pd


# ---------------------- Chargement ---------------------
def reduce_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Réduit la précision numérique (float64 -> float32, int64 -> plus petit entier).

    Indispensable avec 8 Go de RAM : divise environ par 2 l'empreinte mémoire.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Mémoire : {start_mem:.0f} Mo -> {end_mem:.0f} Mo")
    return df


def load_csv(path, num_rows: int | None = None, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=num_rows)
    df = reduce_memory_usage(df)
    if verbose:
        print(f"{path.name} : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    return df


def load_columns_description(path) -> pd.DataFrame:
    # Le fichier de description n'est pas encodé en UTF-8
    return pd.read_csv(path, encoding="latin-1", index_col=0)


# ---------------------- Qualité des données ---------------------
def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre et pourcentage de valeurs manquantes par colonne, triés décroissant."""
    missing = df.isna().sum()
    table = pd.DataFrame(
        {
            "nb_manquants": missing,
            "pct_manquants": 100 * missing / len(df),
            "dtype": df.dtypes.astype(str),
        }
    )
    return table[table["nb_manquants"] > 0].sort_values("pct_manquants", ascending=False)


def data_quality_summary(tables: dict) -> pd.DataFrame:
    """Résumé par table : dimensions, doublons, % global de valeurs manquantes."""
    rows = []
    for name, df in tables.items():
        rows.append(
            {
                "table": name,
                "nb_lignes": df.shape[0],
                "nb_colonnes": df.shape[1],
                "nb_doublons_lignes": int(df.duplicated().sum()),
                "pct_cellules_manquantes": round(100 * df.isna().mean().mean(), 1),
                "memoire_Mo": round(df.memory_usage(deep=True).sum() / 1024**2, 0),
            }
        )
    return pd.DataFrame(rows).set_index("table")


def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    """One-hot encoding des colonnes texte (repris du kernel jsaguiar).

    Retourne le DataFrame encodé et la liste des nouvelles colonnes créées.
    """
    original_columns = list(df.columns)
    categorical_columns = [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]
    df = pd.get_dummies(
        df, columns=categorical_columns, dummy_na=nan_as_category, dtype=np.uint8
    )
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# ---------------------- Nettoyage final ---------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les caractères spéciaux des noms de colonnes.

    LightGBM refuse les noms contenant des caractères JSON (",", ":", "[", ...).
    """
    new_names = [re.sub(r"[^0-9a-zA-Z_]+", "_", col).strip("_") for col in df.columns]
    df.columns = new_names
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Noms de colonnes en double après nettoyage : {duplicated}")
    return df


def replace_infinite_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Les ratios (division par 0) créent des +/-inf : on les remplace par NaN."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[numeric_cols]).sum().sum())
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if verbose:
        print(f"{n_inf:,} valeurs infinies remplacées par NaN")
    return df
