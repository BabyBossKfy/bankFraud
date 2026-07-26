import pandas ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.config import (
    FEATURES,
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def model_catalog() -> dict:
    return {
        "Régression logistique": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=450,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.045,
            max_depth=3,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_models(df: pd.DataFrame, progress=None):
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    results = {}
    catalog = model_catalog()

    for idx, (name, algo) in enumerate(catalog.items(), start=1):
        pipe = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", algo),
        ])

        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.50).astype(int)

        results[name] = {
            "pipe": pipe,
            "pred": pred,
            "proba": proba,
            "metrics": {
                "Accuracy": accuracy_score(y_test, pred),
                "Précision": precision_score(y_test, pred, zero_division=0),
                "Rappel": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, proba),
                "PR-AUC": average_precision_score(y_test, proba),
            },
        }

        if progress is not None:
            progress.progress(idx / len(catalog), text=f"Entraînement : {name}")

    best_name = max(results, key=lambda n: results[n]["metrics"]["PR-AUC"])

    return results, best_name, X_test, y_test


def metrics_frame(models: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"Modèle": name, **obj["metrics"]}
        for name, obj in models.items()
    ])


def fit_ano**ly_detector(df: pd.DataFrame) -> **.DataFrame:
    cols = [
        "transaction_amount",
        "transaction_hour",
        "transactions_last_24h",
        "avg_amount_30d",
        "distance_from_home_km",
        "account_age_days",
    ]**    x = df[cols].copy()
    x = x**illna(x.median(numeric_only=True)**
    scaler = StandardScaler()
  **x_scaled = scaler.fit_transform(x**
    iso = IsolationForest(
     ** contamination=0.04,
        n_es**mators=350,
        random_state=**,
    )

    iso.fit(x_scaled)

 ** raw_score = -iso.score_samples(x**caled)
    span = raw_score.max()** raw_score.min()

    out = df.co**()
    out["score_anomalie"] = 10*** (raw_score - raw_score.min()) /**span if span else 1)
    out["alerte_anomalie"] = (iso.predict(x_sca**d) == -1).astype(int)

    return**ut


def get_feature_names(pipe: **peline) -> list:
    preprocessor** pipe.named_steps["preprocessor"]**    names = []
    names.extend(N**ERIC_FEATURES)

    try:
        **t_pipe = preprocessor.named_trans**rmers_["cat"]
        encoder = c**_pipe.named_steps["onehot"]
     ** encoded_names = encoder.get_feat**e_names_out(CATEGORICAL_FEATURES)**       names.extend(encoded_names**olist())
    except Exception:
  **    names.extend(CATEGORICAL_FEAT**ES)

    return names


def compu**_feature_importance(
    pipe: Pi**line,
    X_test: pd.DataFrame,
 ** y_test: pd.Series,
) -> pd.DataF**me:
    model = pipe.named_steps["model"]

    if hasattr(model, "fe**ure_importances_"):
        names** get_feature_names(pipe)
        **lues = model.feature_importances_**        size = min(len(names), le**values))

        importance = pd**ataFrame({
            "Variable"**names[:size],
            "Import**ce": values[:size],
            "**thode": "Importance native du mod**e",
        })

        return im**rtance.sort_values("Importance", **cending=False)

    result = perm**ation_importance(
        pipe,
 **     X_test,
        y_test,
    **  n_repeats=5,
        random_sta**=42,
        scoring="average_pre**sion",
    )

    importance = pd**ataFrame({
        "Variable": X_**st.columns,
        "Importance":**esult.importances_mean,
        "**thode": "Permutation importance",**   })

    return importance.sort**alues("Importance", ascending=Fal**)


def risk_level(score: float) ** str:
    if score >= 75:
       **eturn "Critique"
    if score >= **:
        return "Élevé"
    if s**re >= 25:
        return "Modéré"**   return "Faible"


def decision**rom_score(score: float, threshold**float) -> str:
    if score >= th**shold:
        return "Fraude sus**ctée"
    return "Transaction nor**le"