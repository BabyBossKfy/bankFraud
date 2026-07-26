from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Détection de la fraude bancaire"
SIGNATURE = "Développé par Franck Kianguebeni - Data Analyst / BI Engineer"

NORMAL_LABEL = "Normal"
SUSPECT_LABEL = "Suspect"
FRAUD_LABEL = "Fraude"


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 22px;
    background: linear-gradient(120deg, #071b33, #0f4c75 65%, #2e9cca);
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 14px 34px rgba(7,27,51,.22);
}

.hero h1 {
    margin: 0;
    font-size: 2.2rem;
}

.hero p {
    margin: .35rem 0 0;
    opacity: .93;
}

.kpi-card {
    padding: 1rem;
    border: 1px solid #dce6f0;
    border-radius: 18px;
    background: white;
    min-height: 115px;
    box-shadow: 0 6px 18px rgba(15,76,117,.06);
}

.kpi-label {
    font-size: .85rem;
    color: #64748b;
}

.kpi-value {
    font-size: 1.65rem;
    font-weight: 780;
    color: #071b33;
    margin: .25rem 0;
}

.kpi-note {
    font-size: .78rem;
    color: #64748b;
}

.section-note {
    padding: .85rem 1rem;
    border-radius: 13px;
    background: #eef6fb;
    color: #071b33;
    border-left: 4px solid #0f4c75;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: .85rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
}

.stButton>button {
    border-radius: 11px;
    font-weight: 650;
}

.stDownloadButton>button {
    border-radius: 11px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce6f0;
    padding: .85rem;
    border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# AFFICHAGE
# ============================================================

def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
    <h1>🏦 Détection de la fraude bancaire</h1>
    <p>Analyse, prédiction, scoring et simulation des transactions bancaires</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def format_number(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


# ============================================================
# DONNÉES
# ============================================================

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        for col in out.columns
    ]
    return out


def find_target_column(df: pd.DataFrame) -> str | None:
    possible_targets = [
        "target",
        "classe",
        "class",
        "label",
        "etat",
        "statut",
        "status",
        "decision",
        "risque",
        "niveau_risque",
        "is_fraud",
        "isfraud",
        "fraud",
        "fraude",
        "churn",
        "exited",
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    return None


def map_target_to_binary(value) -> int:
    text = str(value).strip().lower()

    normal_values = [
        "0",
        "normal",
        "normale",
        "valide",
        "validee",
        "ok",
        "false",
        "no",
        "non",
        "legitime",
        "legitimate",
    ]

    risk_values = [
        "1",
        "2",
        "fraude",
        "fraud",
        "frauduleux",
        "suspect",
        "suspecte",
        "en_attente",
        "en attente",
        "attente",
        "bloque",
        "bloquee",
        "rejetee",
        "chargeback",
        "conteste",
        "true",
        "yes",
        "oui",
    ]

    if text in normal_values:
        return 0

    if text in risk_values:
        return 1

    if "fraud" in text or "fraude" in text or "suspect" in text:
        return 1

    if "bloqu" in text or "rejet" in text or "contest" in text:
        return 1

    if "valid" in text or "normal" in text or "legitime" in text:
        return 0

    return 1


def risk_label_from_score(score: float, seuil_suspect: float, seuil_fraude: float) -> str:
    if score >= seuil_fraude:
        return FRAUD_LABEL
    if score >= seuil_suspect:
        return SUSPECT_LABEL
    return NORMAL_LABEL


def risk_icon(label: str) -> str:
    if label == FRAUD_LABEL:
        return "Rouge"
    if label == SUSPECT_LABEL:
        return "Jaune"
    return "Vert"


@st.cache_data
def create_demo_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    transaction_type = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        size=n,
        p=[0.35, 0.20, 0.25, 0.10, 0.10],
    )

    amount = rng.gamma(shape=2.2, scale=85000, size=n).round(2)

    oldbalanceorg = rng.gamma(shape=2.0, scale=180000, size=n).round(2)
    newbalanceorig = np.maximum(oldbalanceorg - amount, 0).round(2)

    oldbalancedest = rng.gamma(shape=2.0, scale=150000, size=n).round(2)
    newbalancedest = (oldbalancedest + amount).round(2)

    is_transfer_or_cashout = np.isin(transaction_type, ["TRANSFER", "CASH_OUT"]).astype(int)
    empty_origin_after = (newbalanceorig == 0).astype(int)
    high_amount = (amount > np.quantile(amount, 0.90)).astype(int)

    fraud_signal = (
        1.9 * is_transfer_or_cashout
        + 1.6 * empty_origin_after
        + 1.2 * high_amount
        + 0.0000025 * amount
        + rng.normal(0, 1.2, size=n)
    )

    probability = 1 / (1 + np.exp(-(fraud_signal - 4.5)))
    isfraud = rng.binomial(1, probability).astype(int)

    df = pd.DataFrame({
        "step": rng.integers(1, 744, size=n),
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceorg,
        "newbalanceOrig": newbalanceorig,
        "oldbalanceDest": oldbalancedest,
        "newbalanceDest": newbalancedest,
        "isFraud": isfraud,
    })

    return df


def prepare_dataset(raw_df: pd.DataFrame):
    df = normalise_columns(raw_df)

    target_col = find_target_column(df)

    if target_col is None:
        raise ValueError(
            "Aucune colonne cible trouvée. Le fichier doit contenir une colonne comme "
            "isFraud, is_fraud, fraud, fraude, target, status, classe, label ou churn."
        )

    df["target_binary"] = df[target_col].apply(map_target_to_binary).astype(int)

    if df["target_binary"].nunique() < 2:
        raise ValueError(
            "La colonne cible détectée ne contient qu'une seule classe après conversion. "
            "Vérifie que la colonne cible contient bien des valeurs normales et frauduleuses."
        )

    excluded = [
        target_col,
        "target_binary",
        "target_final",
        "prediction_risque",
        "score_risque",
        "niveau_visuel",
    ]

    feature_cols = []

    for col in df.columns:
        if col in excluded:
            continue

        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / max(len(df), 1)

        is_object = df[col].dtype == "object"

        if is_object and unique_ratio > 0.40:
            continue

        if "id" in col.lower():
            continue

        if col.lower() in ["nameorig", "namedest", "name_orig", "name_dest"]:
            continue

        feature_cols.append(col)

    if len(feature_cols) == 0:
        raise ValueError(
            "Aucune variable explicative exploitable n'a été trouvée après nettoyage."
        )

    X = df[feature_cols].copy()
    y = df["target_binary"].copy()

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = [col for col in X.columns if col not in numeric_features]

    return df, target_col, feature_cols, numeric_features, categorical_features, X, y


# ============================================================
# MACHINE LEARNING
# ============================================================

def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_encoder()),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_model_catalog():
    return {
        "Régression logistique": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


@st.cache_resource
def train_models_cached(
    df: pd.DataFrame,
    feature_cols: tuple,
    numeric_features: tuple,
    categorical_features: tuple,
):
    feature_cols = list(feature_cols)
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    X = df[feature_cols].copy()
    y = df["target_binary"].copy()

    if len(df) > 15000:
        sampled_df = df.sample(15000, random_state=42)
        X = sampled_df[feature_cols].copy()
        y = sampled_df["target_binary"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {}
    rows = []

    for model_name, algo in get_model_catalog().items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", algo),
        ])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
        else:
            proba = pred

        rows.append({
            "Modèle": model_name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Rappel": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, proba),
            "PR_AUC": average_precision_score(y_test, proba),
        })

        models[model_name] = {
            "pipeline": pipe,
            "pred": pred,
            "proba": proba,
        }

    scores = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    best_name = scores.iloc[0]["Modèle"]

    return models, scores, best_name, X_test, y_test


def score_all_rows(model_obj, df: pd.DataFrame, feature_cols, seuil_suspect, seuil_fraude):
    X = df[feature_cols].copy()
    pipe = model_obj["pipeline"]

    result = df.copy()

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    else:
        proba = pipe.predict(X)

    result["score_risque"] = (proba * 100).round(2)

    result["prediction_risque"] = result["score_risque"].apply(
        lambda score: risk_label_from_score(score, seuil_suspect, seuil_fraude)
    )

    result["niveau_visuel"] = result["prediction_risque"].apply(risk_icon)

    return result


def get_feature_names(pipe, numeric_features, categorical_features):
    names = []
    names.extend(numeric_features)

    try:
        preprocessor = pipe.named_steps["preprocessor"]

        if "cat" in preprocessor.named_transformers_:
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_names = encoder.get_feature_names_out(categorical_features).tolist()
            names.extend(cat_names)
    except Exception:
        names.extend(categorical_features)

    return names


def compute_feature_importance(model_obj, numeric_features, categorical_features):
    pipe = model_obj["pipeline"]
    model = pipe.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "Variable": numeric_features + categorical_features,
            "Importance": np.zeros(len(numeric_features + categorical_features)),
        })

    names = get_feature_names(pipe, numeric_features, categorical_features)
    values = model.feature_importances_

    size = min(len(names), len(values))

    importance = pd.DataFrame({
        "Variable": names[:size],
        "Importance": values[:size],
    })

    return importance.sort_values("Importance", ascending=False)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Fraude bancaire")
st.sidebar.caption("Normal / Suspect / Fraude")

uploaded_file = st.sidebar.file_uploader(
    "Importer le fichier CSV",
    type=["csv"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Seuils de décision")

seuil_suspect = st.sidebar.slider(
    "Seuil Suspect (%)",
    min_value=5,
    max_value=80,
    value=30,
    step=5,
)

seuil_fraude = st.sidebar.slider(
    "Seuil Fraude (%)",
    min_value=20,
    max_value=95,
    value=70,
    step=5,
)

if seuil_suspect >= seuil_fraude:
    st.sidebar.warning("Le seuil Suspect doit être inférieur au seuil Fraude.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Résumé exécutif",
        "Données",
        "Analyse des variables",
        "Modèles IA",
        "Prédiction par entrée",
        "Simulation transaction",
        "Importance des variables",
        "Export",
    ],
)


# ============================================================
# CHARGEMENT GLOBAL
# ============================================================

render_hero()

try:
    with st.spinner("Chargement des données et entraînement des modèles IA..."):
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            demo_mode = False
        else:
            raw_df = create_demo_data()
            demo_mode = True

        (
            df,
            target_col,
            feature_cols,
            numeric_features,
            categorical_features,
            X,
            y,
        ) = prepare_dataset(raw_df)

        models, scores, best_name, X_test, y_test = train_models_cached(
            df,
            tuple(feature_cols),
            tuple(numeric_features),
            tuple(categorical_features),
        )

        best_model = models[best_name]

        scored_df = score_all_rows(
            best_model,
            df,
            feature_cols,
            seuil_suspect,
            seuil_fraude,
        )

except Exception as exc:
    st.error("Erreur lors du chargement ou de l'entraînement du modèle.")
    st.exception(exc)
    st.stop()


if demo_mode:
    st.info(
        "Aucun fichier CSV importé. L'application utilise un dataset de démonstration "
        "orienté transactions bancaires."
    )


# ============================================================
# PAGE RÉSUMÉ
# ============================================================

if page == "Résumé exécutif":
    st.subheader("Résumé exécutif")

    total = len(scored_df)
    normal_count = int((scored_df["prediction_risque"] == NORMAL_LABEL).sum())
    suspect_count = int((scored_df["prediction_risque"] == SUSPECT_LABEL).sum())
    fraud_count = int((scored_df["prediction_risque"] == FRAUD_LABEL).sum())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi("Transactions analysées", format_number(total), "Volume total")

    with c2:
        render_kpi("Normales", format_number(normal_count), "Risque faible")

    with c3:
        render_kpi("Suspectes", format_number(suspect_count), "Vérification recommandée")

    with c4:
        render_kpi("Fraudes", format_number(fraud_count), "Risque élevé")

    st.write("")

    left, right = st.columns(2)

    class_df = scored_df["prediction_risque"].value_counts().reset_index()
    class_df.columns = ["Classe", "Nombre"]

    left.plotly_chart(
        px.pie(
            class_df,
            names="Classe",
            values="Nombre",
            hole=0.55,
            title="Répartition des prédictions",
        ),
        use_container_width=True,
    )

    right.plotly_chart(
        px.bar(
            scores,
            x="Modèle",
            y="PR_AUC",
            text_auto=".3f",
            range_y=[0, 1],
            title="Classement des modèles selon PR-AUC",
        ),
        use_container_width=True,
    )

    st.subheader("Description des niveaux de risque")
    st.markdown(
        f"""
- **Normal** : score inférieur à `{seuil_suspect} %`.
- **Suspect** : score entre `{seuil_suspect} %` et `{seuil_fraude} %`.
- **Fraude** : score supérieur ou égal à `{seuil_fraude} %`.
"""
    )

    st.subheader("Aperçu des résultats")
    st.dataframe(
        scored_df.head(300),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE DONNÉES
# ============================================================

elif page == "Données":
    st.subheader("Données utilisées")

    c1, c2, c3 = st.columns(3)

    c1.metric("Lignes", len(df))
    c2.metric("Colonnes", len(df.columns))
    c3.metric("Variables modèle", len(feature_cols))

    st.write(f"Colonne cible détectée : `{target_col}`")

    st.subheader("Aperçu du dataset")
    st.dataframe(
        df.head(500),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Variables utilisées par le modèle")
    st.write(feature_cols)

    st.subheader("Structure des colonnes")

    structure = pd.DataFrame({
        "Colonne": df.columns,
        "Type": [str(df[col].dtype) for col in df.columns],
        "Valeurs manquantes": [int(df[col].isna().sum()) for col in df.columns],
        "Valeurs uniques": [int(df[col].nunique()) for col in df.columns],
    })

    st.dataframe(
        structure,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Statistiques descriptives")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True,
    )


# ============================================================
# PAGE ANALYSE DES VARIABLES
# ============================================================

elif page == "Analyse des variables":
    st.subheader("Analyse et comparaison des variables")

    analysis_df = scored_df.copy()

    if numeric_features:
        selected_num = st.selectbox(
            "Variable numérique à analyser",
            numeric_features,
        )

        c1, c2 = st.columns(2)

        c1.plotly_chart(
            px.box(
                analysis_df,
                x="prediction_risque",
                y=selected_num,
                color="prediction_risque",
                title=f"Distribution de {selected_num} par niveau de risque",
            ),
            use_container_width=True,
        )

        grouped = analysis_df.groupby("prediction_risque")[selected_num].agg(
            ["count", "mean", "median", "min", "max", "std"]
        ).reset_index()

        c2.dataframe(
            grouped,
            use_container_width=True,
            hide_index=True,
        )

    if categorical_features:
        selected_cat = st.selectbox(
            "Variable catégorielle à analyser",
            categorical_features,
        )

        cat_df = (
            analysis_df
            .groupby([selected_cat, "prediction_risque"])
            .size()
            .reset_index(name="Nombre")
        )

        st.plotly_chart(
            px.bar(
                cat_df,
                x=selected_cat,
                y="Nombre",
                color="prediction_risque",
                barmode="group",
                title=f"Répartition de {selected_cat} par niveau de risque",
            ),
            use_container_width=True,
        )

    if len(numeric_features) >= 2:
        st.subheader("Matrice de corrélation")

        corr = df[numeric_features].corr(numeric_only=True)

        st.plotly_chart(
            px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Corrélation entre variables numériques",
            ),
            use_container_width=True,
        )


# ============================================================
# PAGE MODÈLES
# ============================================================

elif page == "Modèles IA":
    st.subheader("Comparaison des modèles IA")

    st.markdown(
        """
<div class="section-note">
Chaque modèle apprend à estimer un score de risque. Le meilleur modèle est sélectionné automatiquement selon le PR-AUC.
</div>
""",
        unsafe_allow_html=True,
    )

    st.dataframe(
        scores.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Rappel": "{:.3f}",
            "F1": "{:.3f}",
            "ROC_AUC": "{:.3f}",
            "PR_AUC": "{:.3f}",
        }).highlight_max(axis=0),
        use_container_width=True,
        hide_index=True,
    )

    selected_model = st.selectbox(
        "Inspecter un modèle",
        list(models.keys()),
        index=list(models.keys()).index(best_name),
    )

    model_obj = models[selected_model]

    cm = confusion_matrix(y_test, model_obj["pred"], labels=[0, 1])

    labels_display = ["Normal", "Risque"]

    c1, c2 = st.columns(2)

    c1.plotly_chart(
        px.imshow(
            cm,
            x=labels_display,
            y=labels_display,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Matrice de confusion - {selected_model}",
            labels={"x": "Classe prédite", "y": "Classe réelle"},
        ),
        use_container_width=True,
    )

    fpr, tpr, _ = roc_curve(y_test, model_obj["proba"])
    precision, recall, _ = precision_recall_curve(y_test, model_obj["proba"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Aléatoire",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Courbe ROC - {selected_model}",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
    )

    c2.plotly_chart(fig, use_container_width=True)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, name="Precision-Recall"))
    fig_pr.update_layout(
        title=f"Courbe Precision-Recall - {selected_model}",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )

    st.plotly_chart(fig_pr, use_container_width=True)

    report = classification_report(
        y_test,
        model_obj["pred"],
        target_names=["Normal", "Risque"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    st.subheader("Rapport de classification")
    st.dataframe(report_df, use_container_width=True)


# ============================================================
# PAGE PRÉDICTION PAR ENTRÉE
# ============================================================

elif page == "Prédiction par entrée":
    st.subheader("Prédiction du risque pour chaque entrée")

    st.markdown(
        """
<div class="section-note">
Chaque ligne du fichier reçoit un score de risque puis une décision : Normal, Suspect ou Fraude.
</div>
""",
        unsafe_allow_html=True,
    )

    display_cols = feature_cols[:10] + [
        target_col,
        "score_risque",
        "prediction_risque",
        "niveau_visuel",
    ]

    display_cols = [col for col in display_cols if col in scored_df.columns]

    st.dataframe(
        scored_df[display_cols].head(1000),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Synthèse des prédictions")

    pred_count = scored_df["prediction_risque"].value_counts().reset_index()
    pred_count.columns = ["Niveau", "Nombre"]

    st.plotly_chart(
        px.bar(
            pred_count,
            x="Niveau",
            y="Nombre",
            color="Niveau",
            text_auto=True,
            title="Nombre d'entrées par niveau de risque",
        ),
        use_container_width=True,
    )

    st.subheader("Distribution du score de risque")

    st.plotly_chart(
        px.histogram(
            scored_df,
            x="score_risque",
            color="prediction_risque",
            nbins=50,
            title="Distribution des scores de risque",
        ),
        use_container_width=True,
    )


# ============================================================
# PAGE SIMULATION TRANSACTION
# ============================================================

elif page == "Simulation transaction":
    st.subheader("Simulation d'une nouvelle transaction")

    st.markdown(
        """
<div class="section-note">
Cette section permet de saisir manuellement une nouvelle transaction, puis de calculer automatiquement son score de risque avec le meilleur modèle IA.
</div>
""",
        unsafe_allow_html=True,
    )

    st.success(f"Modèle utilisé : {best_name}")

    pipe = best_model["pipeline"]

    preferred_order = [
        "step",
        "type",
        "amount",
        "oldbalanceorg",
        "newbalanceorig",
        "oldbalancedest",
        "newbalancedest",
    ]

    ordered_features = []

    for col in preferred_order:
        if col in feature_cols:
            ordered_features.append(col)

    for col in feature_cols:
        if col not in ordered_features:
            ordered_features.append(col)

    manual_values = {}

    with st.form("simulation_transaction_form"):
        st.write("Saisis les informations de la transaction à analyser.")

        cols = st.columns(2)

        for idx, col in enumerate(ordered_features):
            container = cols[idx % 2]

            with container:
                label = col.replace("_", " ").capitalize()

                if col in numeric_features:
                    series = pd.to_numeric(df[col], errors="coerce")

                    if series.notna().sum() > 0:
                        default_value = float(series.median())
                        min_value = float(series.min())
                    else:
                        default_value = 0.0
                        min_value = 0.0

                    manual_values[col] = st.number_input(
                        label,
                        min_value=0.0 if min_value >= 0 else min_value,
                        value=default_value,
                        step=1.0,
                        key=f"sim_num_{col}",
                    )

                else:
                    values = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .sort_values()
                        .unique()
                        .tolist()
                    )

                    if len(values) == 0:
                        manual_values[col] = st.text_input(
                            label,
                            value="",
                            key=f"sim_text_{col}",
                        )

                    elif len(values) <= 30:
                        manual_values[col] = st.selectbox(
                            label,
                            values,
                            key=f"sim_cat_{col}",
                        )

                    else:
                        manual_values[col] = st.text_input(
                            label,
                            value=str(values[0]),
                            key=f"sim_text_{col}",
                        )

        submit_simulation = st.form_submit_button(
            "Analyser la transaction",
            use_container_width=True,
        )

    if submit_simulation:
        new_entry = pd.DataFrame([manual_values])
        new_entry = new_entry[feature_cols]

        try:
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                probability = float(pipe.predict_proba(new_entry)[0][1])
            else:
                prediction_binary = int(pipe.predict(new_entry)[0])
                probability = float(prediction_binary)

            score = probability * 100

            final_label = risk_label_from_score(
                score,
                seuil_suspect,
                seuil_fraude,
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Score de risque", f"{score:.2f} %")

            with c2:
                st.metric("Décision", final_label)

            with c3:
                st.metric("Modèle IA", best_name)

            if final_label == FRAUD_LABEL:
                st.error(
                    f"Fraude probable détectée. Score de risque : {score:.2f} %."
                )

            elif final_label == SUSPECT_LABEL:
                st.warning(
                    f"Transaction suspecte. Score de risque : {score:.2f} %. "
                    "Une vérification manuelle est recommandée."
                )

            else:
                st.success(
                    f"Transaction considérée comme normale. Score de risque : {score:.2f} %."
                )

            st.subheader("Détails de la transaction simulée")

            st.dataframe(
                new_entry,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Comparaison avec l'historique")

            comparison_df = scored_df[["score_risque", "prediction_risque"]].copy()
            comparison_df["source"] = "Historique"

            simulated_row = pd.DataFrame({
                "score_risque": [score],
                "prediction_risque": [final_label],
                "source": ["Transaction simulée"],
            })

            comparison_df = pd.concat(
                [comparison_df, simulated_row],
                ignore_index=True,
            )

            fig = px.histogram(
                comparison_df,
                x="score_risque",
                color="source",
                nbins=50,
                title="Comparaison du score simulé avec les scores historiques",
                labels={
                    "score_risque": "Score de risque (%)",
                    "source": "Source",
                },
            )

            fig.add_vline(
                x=score,
                line_dash="dash",
                line_color="red",
                annotation_text="Transaction simulée",
                annotation_position="top right",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Lecture métier du résultat")

            if final_label == FRAUD_LABEL:
                st.markdown(
                    """
- Le score dépasse le seuil de fraude défini.
- La transaction doit être bloquée ou examinée immédiatement.
- Une validation par l'équipe conformité, risque ou opérations est recommandée.
"""
                )

            elif final_label == SUSPECT_LABEL:
                st.markdown(
                    """
- Le score se situe dans la zone intermédiaire.
- La transaction n'est pas automatiquement considérée comme frauduleuse.
- Une analyse complémentaire est recommandée avant validation définitive.
"""
                )

            else:
                st.markdown(
                    """
- Le score est inférieur au seuil suspect.
- La transaction présente un niveau de risque faible selon le modèle.
- Elle peut être considérée comme normale selon les règles actuelles.
"""
                )

        except Exception as exc:
            st.error("Erreur lors de la simulation de la transaction.")
            st.exception(exc)


# ============================================================
# PAGE IMPORTANCE
# ============================================================

elif page == "Importance des variables":
    st.subheader("Importance des variables")

    available_importance_models = [
        model for model in ["Random Forest", "Gradient Boosting", "Extra Trees"]
        if model in models
    ]

    if len(available_importance_models) == 0:
        st.info("Aucun modèle avec importance des variables n'est disponible.")
    else:
        selected_model = st.selectbox(
            "Choisir le modèle",
            available_importance_models,
        )

        importance_df = compute_feature_importance(
            models[selected_model],
            numeric_features,
            categorical_features,
        )

        if len(importance_df) == 0:
            st.info("Aucune importance de variable disponible.")
        else:
            top_n = st.slider(
                "Nombre de variables à afficher",
                min_value=1,
                max_value=min(30, len(importance_df)),
                value=min(15, len(importance_df)),
            )

            st.plotly_chart(
                px.bar(
                    importance_df.head(top_n).sort_values("Importance"),
                    x="Importance",
                    y="Variable",
                    orientation="h",
                    title=f"Top {top_n} variables importantes - {selected_model}",
                ),
                use_container_width=True,
            )

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# PAGE EXPORT
# ============================================================

elif page == "Export":
    st.subheader("Exporter les résultats")

    export_df = scored_df.copy()

    csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.download_button(
        "Télécharger le fichier avec prédictions",
        csv_bytes,
        "resultats_predictions_risque.csv",
        "text/csv",
        use_container_width=True,
    )

    st.success(
        "Le fichier exporté contient les colonnes score_risque, prediction_risque et niveau_visuel."
    )

    st.dataframe(
        export_df.head(500),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
<div class="footer">
    {SIGNATURE}<br>
    Application professionnelle de scoring, classification et simulation du risque bancaire.
</div>
""",
    unsafe_allow_html=True,
)