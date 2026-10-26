import os
from pathlib import Path

# ---------------------- Chemins du projet ---------------------
# Chemins relatifs à la racine du dépôt : le même code tourne sur le Mac et sur la VM.
# DATA_PATH peut être surchargé par variable d'environnement si les données sont ailleurs.
PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_PATH / "data"))
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
FIGURES_PATH = PROJECT_PATH / "reports" / "figures"

# ---------------------- Fichiers bruts (Home Credit) ---------------------
APPLICATION_TRAIN_FILE = RAW_DATA_PATH / "application_train.csv"
APPLICATION_TEST_FILE = RAW_DATA_PATH / "application_test.csv"
BUREAU_FILE = RAW_DATA_PATH / "bureau.csv"
BUREAU_BALANCE_FILE = RAW_DATA_PATH / "bureau_balance.csv"
PREVIOUS_APPLICATION_FILE = RAW_DATA_PATH / "previous_application.csv"
POS_CASH_FILE = RAW_DATA_PATH / "POS_CASH_balance.csv"
INSTALLMENTS_FILE = RAW_DATA_PATH / "installments_payments.csv"
CREDIT_CARD_FILE = RAW_DATA_PATH / "credit_card_balance.csv"
COLUMNS_DESCRIPTION_FILE = RAW_DATA_PATH / "HomeCredit_columns_description.csv"

# ---------------------- Fichiers produits ---------------------
FEATURES_FULL_FILE = PROCESSED_DATA_PATH / "features_full.parquet"
TRAIN_FILE = PROCESSED_DATA_PATH / "train.parquet"
TEST_FILE = PROCESSED_DATA_PATH / "test.parquet"
NEW_CLIENTS_FILE = PROCESSED_DATA_PATH / "new_clients.parquet"
FEATURES_USED_FILE = PROJECT_PATH / "features_used.json"

# ---------------------- MLflow ---------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_PATH / 'mlflow.db'}")

# ---------------------- Colonnes clés ---------------------
ID_COLUMN = "SK_ID_CURR"
TARGET = "TARGET"  # 1 = défaut de paiement, 0 = remboursement

# ---------------------- Paramètres métier et techniques ---------------------
COST_FN = 10  # mauvais client prédit bon : crédit accordé, perte en capital
COST_FP = 1  # bon client prédit mauvais : crédit refusé, manque à gagner
TEST_SIZE = 0.2
random_state = 42
