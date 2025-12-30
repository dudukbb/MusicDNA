# metrics_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics import pairwise_distances

from artifacts import load_artifacts
from config import ARTIFACT_PATH


def _get_embedding(df_final: pd.DataFrame, kmeans):
    """
    Silhouette ve diğer metrikler için 'X' gerekir.
    Biz burada PCA uzayını kullanacağız.
    pipeline.py zaten df_final içine pca1, pca2 yazıyor.
    """
    if "pca1" in df_final.columns and "pca2" in df_final.columns:
        X = df_final[["pca1", "pca2"]].to_numpy(dtype=float)
    else:
        # Eğer pca1/pca2 yoksa, kmeans'in çalıştığı uzayı bulmak zor olur.
        # Bu durumda pipeline'a pca bileşenlerini df'e eklemek gerekir.
        raise ValueError("df_final içinde pca1/pca2 yok. pipeline.py'de pca1/pca2 eklenmiş olmalı.")

    labels = df_final["cluster"].to_numpy(dtype=int)
    return X, labels


def cluster_distance_report(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> dict:
    """
    Küme-içi ve küme-dışı mesafeleri hesaplar.
    - Intra: noktanın kendi centroid'ine uzaklığı
    - Inter: centroidler arası uzaklıklar (min/mean/max)
    """
    # Her noktanın kendi centroid'ine uzaklığı
    intra = np.linalg.norm(X - centroids[labels], axis=1)
    intra_mean = float(np.mean(intra))
    intra_median = float(np.median(intra))

    # Centroidler arası mesafeler
    cdist = pairwise_distances(centroids)
    # diagonal hariç
    mask = ~np.eye(cdist.shape[0], dtype=bool)
    inter_vals = cdist[mask]

    out = {
        "intra_mean": intra_mean,
        "intra_median": intra_median,
        "inter_min": float(np.min(inter_vals)) if inter_vals.size else 0.0,
        "inter_mean": float(np.mean(inter_vals)) if inter_vals.size else 0.0,
        "inter_max": float(np.max(inter_vals)) if inter_vals.size else 0.0,
        "separation_ratio_interMean_over_intraMean": float(np.mean(inter_vals) / (intra_mean + 1e-9))
        if inter_vals.size else 0.0,
    }
    return out


def main():
    art = load_artifacts(ARTIFACT_PATH)
    df_final = art["df_final"]
    kmeans = art["kmeans"]

    X, labels = _get_embedding(df_final, kmeans)
    centroids = kmeans.cluster_centers_

    # KMeans kalite göstergeleri
    sil = float(silhouette_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    inertia = float(kmeans.inertia_)  # SSD (kmeans fit edildiği uzay için)

    dist = cluster_distance_report(X, labels, centroids)

    print("\n===== CLUSTER EVALUATION (PCA 2D uzayı) =====")
    print(f"K (cluster sayısı): {len(np.unique(labels))}")
    print(f"Silhouette score:   {sil:.4f}  (yüksek daha iyi)")
    print(f"Calinski-Harabasz:  {ch:.2f}   (yüksek daha iyi)")
    print(f"Davies-Bouldin:     {db:.4f}  (düşük daha iyi)")
    print(f"Inertia (SSD):      {inertia:.2f}  (düşük daha iyi, ama K artınca düşer)")

    print("\n===== DISTANCE SUMMARY =====")
    print(f"Intra mean (point→own centroid): {dist['intra_mean']:.4f}")
    print(f"Intra median:                    {dist['intra_median']:.4f}")
    print(f"Inter min (centroid↔centroid):   {dist['inter_min']:.4f}")
    print(f"Inter mean:                      {dist['inter_mean']:.4f}")
    print(f"Inter max:                       {dist['inter_max']:.4f}")
    print(f"InterMean / IntraMean ratio:     {dist['separation_ratio_interMean_over_intraMean']:.4f}")

    # Küme bazında intra istatistikleri
    print("\n===== PER-CLUSTER INTRA (mean distance to centroid) =====")
    intra = np.linalg.norm(X - centroids[labels], axis=1)
    tmp = pd.DataFrame({"cluster": labels, "intra": intra})
    per = tmp.groupby("cluster")["intra"].agg(["count", "mean", "median", "std"]).reset_index()
    print(per.to_string(index=False))


if __name__ == "__main__":
    main()
