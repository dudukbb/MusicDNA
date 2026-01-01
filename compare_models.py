# compare_models.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


KMEANS_ARTIFACT = "musicdna_artifacts.joblib"
GMM_ARTIFACT = "musicdna_gmm_artifacts.joblib"


def _ensure_pca2d(art: dict, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    X: PCA 2D (pca1,pca2)
    y: labels
    Eğer df_final içinde pca1/pca2 yoksa, scaler+pca ile yeniden üretir.
    """
    df = art["df_final"].copy()

    if label_col not in df.columns:
        raise ValueError(f"df_final içinde '{label_col}' yok. Önce ilgili pipeline çalıştırılmış olmalı.")

    # PCA kolonları varsa direkt al
    if {"pca1", "pca2"}.issubset(df.columns):
        X2 = df[["pca1", "pca2"]].to_numpy(dtype=float)
    else:
        # yoksa scaler+pca ile üret
        scaler = art["scaler"]
        pca = art["pca"]
        feats = art["model_features"]
        X = df[feats].to_numpy(dtype=float)
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        X2 = X_pca[:, :2]

    y = df[label_col].to_numpy(dtype=int)
    return X2, y


def _metrics(X: np.ndarray, y: np.ndarray) -> dict:
    return {
        "silhouette": float(silhouette_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
    }


def _scatter(X: np.ndarray, y: np.ndarray, title: str):
    plt.figure(figsize=(9, 6))
    for c in np.unique(y):
        m = y == c
        plt.scatter(X[m, 0], X[m, 1], s=8, alpha=0.4, label=f"Cluster {c}")
    plt.xlabel("pca1")
    plt.ylabel("pca2")
    plt.title(title)
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    k_art = joblib.load(KMEANS_ARTIFACT)
    g_art = joblib.load(GMM_ARTIFACT)

    # PCA 2D + label
    Xk, yk = _ensure_pca2d(k_art, "cluster")
    Xg, yg = _ensure_pca2d(g_art, "gmm_cluster")

    mk = _metrics(Xk, yk)
    mg = _metrics(Xg, yg)

    # Tablo
    comp = pd.DataFrame(
        [
            {"model": "KMeans", **mk},
            {"model": "GMM (hard label)", **mg},
        ]
    )
    print("\n===== KMeans vs GMM (PCA 2D) =====")
    print(comp.to_string(index=False))

    # Grafik 1: PCA scatter (KMeans)
    _scatter(Xk, yk, "KMeans — PCA 2D Cluster Map")

    # Grafik 2: PCA scatter (GMM)
    _scatter(Xg, yg, "GMM — PCA 2D Cluster Map (hard label)")

    # Grafik 3: Silhouette bar (karşılaştırma)
    plt.figure(figsize=(6, 4))
    plt.bar(["KMeans", "GMM"], [mk["silhouette"], mg["silhouette"]])
    plt.ylabel("Silhouette score (PCA 2D)")
    plt.title("Silhouette Comparison")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    # Grafik 4: GMM confidence histogram (varsa)
    df_g = g_art["df_final"]
    if "gmm_confidence" in df_g.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(df_g["gmm_confidence"].to_numpy(dtype=float), bins=30)
        plt.xlabel("GMM confidence (max probability)")
        plt.ylabel("count")
        plt.title("GMM Confidence Distribution")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("\n[INFO] GMM confidence kolonları yok. gmm_pipeline.py çalıştırırken df_final'e eklenmiş olmalı.")

    print("\n✅ Bitti. İstersen bu çıktıları rapora 'KMeans vs GMM Ablation' diye koyabiliriz.")


if __name__ == "__main__":
    main()
