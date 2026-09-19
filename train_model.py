"""
Entraînement du modèle de scoring de résiliation (assurance auto).

Usage :
    python train_model.py

Ce script regroupe les parties A, B et C du TP :
  A. chargement des données et choix des variables (sans fuite de données) ;
  B. Pipeline scikit-learn (prétraitement + modèle), validation croisée, évaluation ;
  C. sauvegarde du pipeline (.pkl) et des métadonnées (.json) lues par l'application Streamlit.
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Chemins relatifs à ce fichier : le script fonctionne quel que soit le dossier de lancement.
BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "data" / "dataset_assurance_ML.xlsx"
CSV_PATH = BASE_DIR / "data" / "dataset_assurance_ML.csv"
MODEL_PATH = BASE_DIR / "models" / "pipeline_resiliation.pkl"
META_PATH = BASE_DIR / "models" / "metadata.json"

# ---------------------------------------------------------------------------
# Partie A - Variables
# ---------------------------------------------------------------------------
TARGET = "Résiliation"

# 12 variables d'entrée. Sont exclues : les colonnes d'identité (aucune valeur prédictive)
# et « Statut Contrat », qui révèle la cible à 100 % (fuite de données).
NUM_COLS = [
    "Âge", "Salaire Annuel (€)", "Prime Annuelle (€)", "Ancienneté (mois)",
    "Coeff. Bonus-Malus", "Nb Sinistres (3 ans)",
    "Montant Sinistres (€)", "Score Risque (0-100)",
]
CAT_COLS = ["Type Contrat", "Catégorie Prof.", "Usage Véhicule", "Dernier Sinistre"]


def charger_donnees(ecrire_csv=True):
    """Charge l'Excel, l'enregistre en CSV (utf-8-sig) puis le recharge (étape 1 du TP)."""
    df = pd.read_excel(XLSX_PATH)
    if ecrire_csv:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# Partie B - Pipeline, entraînement, évaluation
# ---------------------------------------------------------------------------
def construire_pipeline(algo):
    """Un seul objet : prétraitement (StandardScaler + OneHotEncoder) puis modèle."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("prep", preprocessor), ("model", algo)])


def entrainer(verbose=False, ecrire_csv=False):
    """Entraîne le pipeline retenu (Random Forest) et renvoie (pipeline, meta)."""
    df = charger_donnees(ecrire_csv=ecrire_csv)
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Classes déséquilibrées (90 / 10) : class_weight='balanced' pour les deux candidats.
    pipelines = {
        "Régression Logistique": construire_pipeline(LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42)),
        "Random Forest": construire_pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=10,
            class_weight="balanced", random_state=42)),
    }

    if verbose:
        print(f"Données : {X.shape[0]} clients, {X.shape[1]} variables, "
              f"taux de résiliation {y.mean():.0%}")
        print(f"Train {X_train.shape} / Test {X_test.shape}")
        print("Validation croisée 5-fold (ROC-AUC, sur le train) :")
        for nom, pipe in pipelines.items():
            scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
            print(f"  {nom:22s} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # Modèle retenu : Random Forest (écart non significatif avec la régression logistique,
    # mais il fournit les importances de variables affichées dans l'interface).
    pipeline = pipelines["Random Forest"]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_test = round(float(roc_auc_score(y_test, y_proba)), 3)

    if verbose:
        print("\nÉvaluation sur le jeu de test :")
        print("  Accuracy :", round(accuracy_score(y_test, y_pred), 3))
        print("  F1       :", round(f1_score(y_test, y_pred), 3))
        print("  ROC-AUC  :", auc_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=["Reste", "Résilie"]))

    # Métadonnées : ce dont l'interface a besoin (colonnes, bornes des curseurs, modalités des menus).
    meta = {
        "modele": "Random Forest",
        "auc_test": auc_test,
        "sklearn_version": sklearn.__version__,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "num_ranges": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                           "median": float(X[c].median())} for c in NUM_COLS},
        "cat_values": {c: sorted(X[c].unique().tolist()) for c in CAT_COLS},
    }
    return pipeline, meta


# ---------------------------------------------------------------------------
# Partie C - Sauvegarde
# ---------------------------------------------------------------------------
def sauvegarder(pipeline, meta):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    pipeline, meta = entrainer(verbose=True, ecrire_csv=True)
    sauvegarder(pipeline, meta)
    taille = MODEL_PATH.stat().st_size / 1024
    print(f"\nPipeline sauvegardé : {MODEL_PATH.relative_to(BASE_DIR)} ({taille:.0f} Ko)")
    print(f"Métadonnées         : {META_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
