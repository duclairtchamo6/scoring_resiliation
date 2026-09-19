# 🚗 Scoring de résiliation — Assurance Auto

Application web qui estime la probabilité de résiliation d'un client d'assurance auto à partir de son profil : Pipeline scikit-learn (Random Forest) et interface Streamlit.

## Comment ça marche

- **Données** : 500 clients, 12 variables retenues (8 numériques, 4 catégorielles). La colonne `Statut Contrat` est exclue, car elle révèle la cible (fuite de données).
- **Modèle** : un `Pipeline` scikit-learn (`StandardScaler` + `OneHotEncoder`, puis Random Forest avec `class_weight='balanced'`), sauvegardé avec joblib.
- **Évaluation** : validation croisée 5-fold puis jeu de test. Seuls 10 % des clients résilient, donc on juge le modèle au ROC-AUC (0,853 sur le test) et non à l'accuracy seule.
- **Interface** : curseurs et menus alimentés par `models/metadata.json`, probabilité de résiliation, niveau de risque et variables les plus influentes.

## Structure du projet

```
scoring_resiliation/
├── data/
│   ├── dataset_assurance_ML.xlsx    # données brutes (500 clients x 27 colonnes)
│   └── dataset_assurance_ML.csv     # même fichier converti en CSV (utf-8-sig)
├── models/
│   ├── pipeline_resiliation.pkl     # pipeline complet (prétraitement + modèle)
│   └── metadata.json                # colonnes, bornes des curseurs, modalités des menus
├── .streamlit/config.toml           # thème de l'application
├── train_model.py                   # entraînement : crée le .pkl et le .json
├── app.py                           # interface Streamlit
├── requirements.txt                 # dépendances figées
└── README.md
```

## Lancer l'application en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Ré-entraîner le modèle

```bash
python train_model.py
```

Le script recrée `models/pipeline_resiliation.pkl` et `models/metadata.json`, et affiche les métriques.

## Déploiement

L'application est prête pour [Streamlit Community Cloud](https://share.streamlit.io) : dépôt GitHub public, branche `main`, fichier principal `app.py`. Les versions des bibliothèques sont figées dans `requirements.txt` pour que le `.pkl` se charge à l'identique sur le serveur.
