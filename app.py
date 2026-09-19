"""
Application Streamlit - scoring de résiliation (assurance auto).

Lancer en local :
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

# Doit être le PREMIER appel Streamlit.
st.set_page_config(page_title="Scoring Résiliation", page_icon="🚗", layout="wide")

# Chemins relatifs à ce fichier : identiques en local et sur Streamlit Community Cloud.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pipeline_resiliation.pkl"
META_PATH = BASE_DIR / "models" / "metadata.json"

SEUIL_RISQUE = 0.55
SEUIL_MODERE = 0.40


@st.cache_resource
def charger_modele():
    """Charge le pipeline et ses métadonnées une seule fois (pas à chaque clic)."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("sklearn_version", sklearn.__version__) != sklearn.__version__:
            raise RuntimeError("Version de scikit-learn différente de celle de l'entraînement.")
    except Exception:
        # Filet de sécurité : .pkl absent ou incompatible avec la version installée
        # sur le serveur -> on ré-entraîne à partir des données (quelques secondes).
        from train_model import entrainer
        pipeline, meta = entrainer()
    return pipeline, meta


pipeline, meta = charger_modele()
num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
rng, cats = meta["num_ranges"], meta["cat_values"]

st.title("🚗 Scoring de résiliation — Assurance Auto")
st.caption(f"Modèle : {meta['modele']} · AUC test : {meta['auc_test']} · "
           f"{len(num_cols) + len(cat_cols)} variables d'entrée")
st.write("Renseignez le profil du client dans le panneau de gauche, puis cliquez sur "
         "**Prédire** pour estimer son risque de résiliation.")

# ---------------------------------------------------------------------------
# Barre latérale : profil du client
# ---------------------------------------------------------------------------
st.sidebar.header("👤 Profil du client")


def curseur(col, step=1.0, fmt=None):
    """Curseur borné par le min/max du metadata et positionné sur la médiane."""
    r = rng[col]
    val = st.sidebar.slider(col, min_value=r["min"], max_value=r["max"],
                            value=r["median"], step=step, format=fmt)
    return int(val) if fmt == "%d" else val


client = {}
client["Âge"] = curseur("Âge", 1.0, "%d")
client["Salaire Annuel (€)"] = curseur("Salaire Annuel (€)", 500.0, "%d")
client["Prime Annuelle (€)"] = curseur("Prime Annuelle (€)", 10.0, "%d")
client["Ancienneté (mois)"] = curseur("Ancienneté (mois)", 1.0, "%d")
client["Coeff. Bonus-Malus"] = curseur("Coeff. Bonus-Malus", 0.01, "%.2f")
client["Nb Sinistres (3 ans)"] = curseur("Nb Sinistres (3 ans)", 1.0, "%d")
client["Montant Sinistres (€)"] = curseur("Montant Sinistres (€)", 100.0, "%d")
client["Score Risque (0-100)"] = curseur("Score Risque (0-100)", 1.0, "%d")

st.sidebar.markdown("---")
# Les modalités viennent du metadata : si on ré-entraîne avec de nouvelles
# catégories, l'interface se met à jour sans modifier app.py.
for col in cat_cols:
    client[col] = st.sidebar.selectbox(col, cats[col])

# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------
if st.button("🔮 Prédire", type="primary", width="stretch"):
    # Le pipeline attend EXACTEMENT les 12 colonnes, dans l'ordre d'entraînement.
    df_client = pd.DataFrame([client])[num_cols + cat_cols]
    proba = float(pipeline.predict_proba(df_client)[0, 1])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Probabilité de résiliation", f"{proba:.0%}")
        if proba >= SEUIL_RISQUE:
            st.error("⚠️ Client À RISQUE — action de rétention conseillée")
        elif proba >= SEUIL_MODERE:
            st.warning("🟠 Risque modéré — à surveiller")
        else:
            st.success("✅ Client fidèle — risque faible")
    with col2:
        st.write("Niveau de risque")
        st.progress(proba)
        st.write("Données envoyées au modèle :")
        # .astype(str) : une colonne qui mélange nombres et textes ferait planter l'affichage.
        st.dataframe(df_client.T.astype(str).rename(columns={0: "Valeur"}),
                     width="stretch")

    # Explication : importance des variables du Random Forest.
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        noms = pipeline.named_steps["prep"].get_feature_names_out()
        imp = (pd.Series(model.feature_importances_, index=noms)
               .sort_values(ascending=False).head(8))
        imp.index = [n.split("__", 1)[1] for n in imp.index]
        st.subheader("📊 Les 8 variables les plus influentes du modèle")
        st.bar_chart(imp)
else:
    st.info("👈 Ajustez le profil dans la barre latérale, puis cliquez sur Prédire.")
