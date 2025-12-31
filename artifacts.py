# artifacts.py
from __future__ import annotations

import joblib


def export_artifacts(
    df_final,
    scaler,
    pca,
    kmeans,
    model_features,
    cluster_names,
    out_path: str,
) -> None:
    payload = {
        "df_final": df_final,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "model_features": model_features,
        "cluster_names": cluster_names,
    }
    joblib.dump(payload, out_path)
    print(f" Kaydedildi: {out_path}")


def load_artifacts(path: str):
    return joblib.load(path)

