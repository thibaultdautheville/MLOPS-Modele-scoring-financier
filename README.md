# MLOPS-Modele-scoring-financier

Projet 6 OpenClassrooms (parcours Data Scientist Machine Learning) : outil de **scoring crédit** pour la société « Prêt à dépenser », données [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data).

## Missions

- Construire et optimiser un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un souci de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.
- Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à la pré-production du modèle (MLflow : tracking, UI, model registry, serving).

Contrainte métier : un faux négatif (crédit accordé à un mauvais payeur) coûte **10 fois** plus qu'un faux positif. Le seuil de décision est optimisé sur ce coût métier.

## Organisation du dépôt

```
├── data/                         # NON versionné (.gitignore) : à copier en local
│   ├── raw/                      #   les 10 CSV Home Credit
│   └── processed/                #   produits par le notebook 01_2 (train/test/new_clients .parquet)
├── notebooks/
│   ├── 01_1_exploration.ipynb            # Étape 1 (1/2) : analyse exploratoire
│   ├── 01_2_preparation_features.ipynb   # Étape 1 (2/2) : jointures, feature engineering, split
│   └── 02_1_mlflow_setup.ipynb           # Étape 2 : configuration MLflow, premier run (baseline)
├── src/
│   ├── settings.py                       # chemins, constantes, paramètres métier
│   ├── data_processing_functions.py      # chargement, qualité, nettoyage
│   ├── feature_engineering_functions.py  # kernel jsaguiar adapté
│   └── modeling_functions.py             # validation croisée, métriques
├── mlflow.db / mlartifacts/      # NON versionnés : runs et artefacts MLflow locaux
├── reports/figures/              # graphiques générés par les notebooks
├── features_used.json            # liste des features retenues (sortie du notebook 01_2)
├── pyproject.toml / uv.lock      # environnement (uv, Mac)
└── requirements.txt              # même environnement pour pip (VM)
```

Les chemins sont tous relatifs à la racine du dépôt (voir `src/settings.py`) : le même code tourne sur le Mac et sur la VM. Si les données sont rangées ailleurs, définir la variable d'environnement `DATA_PATH`.

## Installation

### Mac (avec uv, comme dans le cours *Initiez-vous au ML Engineering*)

```bash
git clone https://github.com/thibaultdautheville/MLOPS-Modele-scoring-financier.git
cd MLOPS-Modele-scoring-financier
uv sync                      # crée .venv avec Python 3.12 et toutes les dépendances
```

### Machine virtuelle (avec pip, sans uv)

```bash
git clone https://github.com/thibaultdautheville/MLOPS-Modele-scoring-financier.git
cd MLOPS-Modele-scoring-financier
python -m venv .venv
# Windows : .venv\Scripts\activate    |    Linux / Mac : source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 à 3.13 est requis (le projet est développé en 3.12).

### Données

Copier les 10 fichiers CSV (dossier `Data` du OneDrive du projet, ou [lien OpenClassrooms](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip)) dans `data/raw/`, puis exécuter `notebooks/01_2_preparation_features.ipynb` pour générer `data/processed/` (environ 1 minute).

Dans VS Code, choisir le noyau `.venv` du projet pour exécuter les notebooks.

### Suivi des expériences (MLflow)

Exécuter `notebooks/02_1_mlflow_setup.ipynb` (création de l'expérience `scoring_credit`), puis, depuis la racine du dépôt :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # Mac
mlflow ui --backend-store-uri sqlite:///mlflow.db            # VM (.venv activé)
```

et ouvrir http://127.0.0.1:5000.

## Routine de travail entre les deux machines

```bash
git pull                     # en début de session
# ... travail ...
git add -A && git commit -m "message" && git push    # en fin de session
```

## Sources méthodologiques

- Cours OpenClassrooms *Maîtriser l'apprentissage supervisé* et *Initiez-vous au ML Engineering* (scripts des screencasts).
- Kernels Kaggle : [Start Here: A Gentle Introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook) (EDA) et [LightGBM with Simple Features](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script) (feature engineering).
- Documentation : [MLflow](https://mlflow.org/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [XGBoost](https://xgboost.readthedocs.io/en/stable/), [Optuna](https://optuna.readthedocs.io/en/stable/), [missingno](https://github.com/ResidentMario/missingno).

## Avancement

- [x] Étape 1 : préparation, nettoyage et enrichissement des données
- [ ] Étape 2 : tracking des expérimentations avec MLflow
- [ ] Étape 3 : modélisation et comparaison de plusieurs algorithmes (validation croisée, déséquilibre)
- [ ] Étape 4 : optimisation des hyperparamètres et du seuil métier
- [ ] Explicabilité globale et locale (SHAP)
- [ ] Model registry et serving MLflow
