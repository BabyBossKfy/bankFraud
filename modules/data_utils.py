import numpy as np
import pandas as pd

from modules.config import (
    DEFAULT_DATA,
    FEATURES,
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        for c in out.columns
    ]
    return out


def generate_demo_data(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    countries = np.array([
        "Congo",
        "France",
        "Nigeria",
        "Cameroun",
        "Maroc",
        "Afrique du Sud",
        "Kenya",
    ])

    merchant_categories = np.array([
        "Alimentation",
        "Carburant",
        "E-commerce",
        "Voyage",
        "Luxe",
        "Transfert",
        "Jeux",
    ])

    payment_channels = np.array([
        "Carte",
        "Mobile Money",
        "Virement",
        "POS",
        "Web",
    ])

    devices = np.array([
        "Mobile",
        "Desktop",
        "POS",
        "ATM",
    ])

    df = pd.DataFrame({
        "transaction_id": [f"TX-{100000 + i}" for i in range(n)],
        "customer_id": [f"CUST-{rng.integers(1000, 9999)}" for _ in range(n)],
        "transaction_amount": rng.gamma(shape=2.2, scale=85, size=n).round(2),
        "transaction_hour": rng.integers(0, 24, size=n),
        "day_of_week": rng.integers(0, 7, size=n),
        "merchant_category": rng.choice(
            merchant_categories,
            size=n,
            p=[0.18, 0.13, 0.25, 0.11, 0.08, 0.18, 0.07],
        ),
        "payment_channel": rng.choice(
            payment_channels,
            size=n,
            p=[0.34, 0.26, 0.13, 0.17, 0.10],
        ),
        "transaction_country": rng.choice(
            countries,
            size=n,
            p=[0.48, 0.13, 0.10, 0.09, 0.07, 0.07, 0.06],
        ),
        "device_type": rng.choice(
            devices,
            size=n,
            p=[0.55, 0.22, 0.16, 0.07],
        ),
        "card_present": rng.choice([0, 1], size=n, p=[0.58, 0.42]),
        "customer_age": rng.integers(18, 79, size=n),
        "account_age_days": rng.integers(10, 4200, size=n),
        "transactions_last_24h": rng.poisson(lam=2.2, size=n),
        "avg_amount_30d": rng.gamma(shape=2.0, scale=60, size=n).round(2),
        "distance_from_home_km": rng.gamma(shape=1.6, scale=35, size=n).round(2),
    })

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["transaction_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_new_device"] = rng.choice([0, 1], size=n, p=[0.86, 0.14])
    df["is_international"] = (df["transaction_country"] != "Congo").astype(int)

    amount_ratio = df["transaction_amount"] / (df["avg_amount_30d"] + 1)

    fraud_signal = (
        0.014 * df["transaction_amount"]
        + 14 * df["is_night"]
        + 18 * df["is_new_device"]
        + 15 * df["is_international"]
        + 10 * (df["transactions_last_24h"] >= 5).astype(int)
        + 9 * (df["distance_from_home_km"] > 120).astype(int)
        + 10 * (amount_ratio > 4).astype(int)
        + 8 * df["merchant_category"].isin(["Luxe", "Jeux", "Transfert"]).astype(int)
        + rng.normal(0, 10, size=n)
    )

    probability = 1 / (1 + np.exp(-(fraud_signal - 55) / 12))
    df["is_fraud"] = rng.binomial(1, probability).astype(int)

    return df


def load_data(uploaded_file=None) -> tuple[pd.DataFrame, bool]:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return normalise_columns(df), False

    if DEFAULT_DATA.exists():
        df = pd.read_csv(DEFAULT_DATA)
        return normalise_columns(df), False

    return generate_demo_data(), True

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    out = normalise_columns(df)

    # Vérification des colonnes obligatoires
    missing = set(FEATURES + [TARGET]) - set(out.columns)

    if missing:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(sorted(missing))
        )

    # Conversion des colonnes numériques
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Conversion des colonnes catégorielles
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).fillna("Inconnu")

    # Nettoyage
    out = out.drop_duplicates()
    out = out.dropna(subset=FEATURES + [TARGET])

    # Conversion de la cible
    out[TARGET] = out[TARGET].astype(int)

    # Création des identifiants si absents
    if "transaction_id" not in out.columns:
        out["transaction_id"] = [
            f"TX-{i+1}" for i in range(len(out))
        ]

    if "customer_id" not in out.columns:
        out["customer_id"] = [
            f"CUST-{i+1}" for i in range(len(out))
        ]

    return out