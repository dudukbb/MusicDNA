# viz_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_samples
from sklearn.metrics import pairwise_distances

from artifacts import load_artifacts
from config import ARTIFACT_PATH


def silhouette_plot(X: np.ndarray, labels: np.ndarray):
    """
    Klasik silhouette plot (kümelerin iç tutarlılık + ayrışma durumunu görselleştirir).
    """
    sil_vals = silhouette_samples(X, labels)
    n_clusters = len(np.unique(labels))

    y_lower = 10
    plt.figure(figsize=(10, 6))

    for c in range(n_clusters):
        c_sil = sil_vals[labels == c]
        c_sil.sort()

        size = c_sil.shape[0]
        y_upper = y_lower + size

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            c_sil,
            alpha=0.7,
        )
        plt.text(-0.05, y_lower + 0.5 * size, f"Cluster {c}")
        y_lower = y_upper + 10

    plt.axvline(np.mean(sil_vals), linestyle="--")
    plt.xlabel("Silhouette value")
    plt.ylabel("Cluster / samples")
    plt.title("Silhouette Plot (PCA 2D uzayı)")
    plt.tight_layout()
    plt.show()


def pca_scatter_with_centroids(df_final: pd.DataFrame, centroids: np.ndarray):
    """
    PCA 2D dağılımı + centroid işaretleri
    """
    plt.figure(figsize=(10, 7))
    for c in sorted(df_final["cluster"].unique()):
        m = df_final["cluster"] == c
        plt.scatter(df_final.loc[m, "pca1"], df_final.loc[m, "pca2"], s=8, alpha=0.4, label=f"Cluster {c}")

    plt.scatter(centroids[:, 0], centroids[:, 1], s=140, marker="x", linewidths=3, label="Centroids")
    plt.xlabel("pca1")
    plt.ylabel("pca2")
    plt.title("PCA 2D Cluster Map")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def centroid_distance_heatmap(centroids: np.ndarray):
    """
    Centroidler arası uzaklık matrisi (inter-cluster separation için güzel görsel).
    """
    D = pairwise_distances(centroids)

    plt.figure(figsize=(7, 6))
    plt.imshow(D)  # renk belirtmiyoruz
    plt.colorbar()
    plt.title("Centroid Distance Matrix (Inter-Cluster)")
    plt.xlabel("Cluster id")
    plt.ylabel("Cluster id")
    plt.tight_layout()
    plt.show()


def main():
    art = load_artifacts(ARTIFACT_PATH)
    df_final = art["df_final"]
    kmeans = art["kmeans"]

    if not {"pca1", "pca2", "cluster"}.issubset(df_final.columns):
        raise ValueError("df_final içinde pca1/pca2/cluster yok. pipeline.py çıktısını kontrol et.")

    X = df_final[["pca1", "pca2"]].to_numpy(dtype=float)
    labels = df_final["cluster"].to_numpy(dtype=int)
    centroids = kmeans.cluster_centers_

    silhouette_plot(X, labels)
    pca_scatter_with_centroids(df_final, centroids)
    centroid_distance_heatmap(centroids)


if __name__ == "__main__":
    main()
