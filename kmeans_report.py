from __future__ import annotations

import sys
from pathlib import Path

# Proje root (bu dosya root'ta olacak)
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics import pairwise_distances

from artifacts import load_artifacts
from config import ARTIFACT_PATH


OUT_DIR = Path("outputs/main")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_table_png(pivot_pct: pd.DataFrame, out_path: Path, title: str) -> None:
    """Pandas pivot tabloyu PNG olarak kaydeder (matplotlib table)."""
    n_rows, n_cols = pivot_pct.shape
    fig_w = max(10, 1.1 * n_cols + 4)
    fig_h = max(3.5, 0.6 * n_rows + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    ax.table(
        cellText=pivot_pct.values,
        rowLabels=[str(r) for r in pivot_pct.index],
        colLabels=[str(c) for c in pivot_pct.columns],
        cellLoc="center",
        loc="center",
    )

    plt.title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    art = load_artifacts(ARTIFACT_PATH)
    df = art["df_final"]
    kmeans = art["kmeans"]

    # PCA 2D uzayı (df_final içinde var)
    if not {"pca1", "pca2", "cluster"}.issubset(df.columns):
        raise ValueError("df_final içinde pca1/pca2/cluster yok. Önce `python pipeline.py` çalıştır.")

    X2 = df[["pca1", "pca2"]].to_numpy(float)
    labels = df["cluster"].to_numpy(int)

    # KMeans PCA uzayında fit edildi => merkezler PCA boyutunda
    centroids_2d = kmeans.cluster_centers_[:, :2]

    # =======================
    # METRICS (PCA 2D)
    # =======================
    sil = float(silhouette_score(X2, labels))
    ch = float(calinski_harabasz_score(X2, labels))
    db = float(davies_bouldin_score(X2, labels))
    inertia = float(getattr(kmeans, "inertia_", float("nan")))

    print("\n===== MAIN CLUSTER REPORT (PCA 2D) =====")
    print(f"K: {len(np.unique(labels))}")
    print(f"Silhouette:        {sil:.4f}  (yüksek daha iyi)")
    print(f"Calinski-Harabasz: {ch:.2f}   (yüksek daha iyi)")
    print(f"Davies-Bouldin:    {db:.4f}  (düşük daha iyi)")
    print(f"Inertia (SSD):     {inertia:.2f}  (K artınca düşer)")

    # =======================
    # SILHOUETTE HIST (PNG)
    # =======================
    sil_vals = silhouette_samples(X2, labels)
    plt.figure(figsize=(9, 4.5))
    plt.hist(sil_vals, bins=40)
    plt.title("Silhouette Distribution (Main Clustering, PCA 2D)")
    plt.xlabel("Silhouette value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "silhouette_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =======================
    # PCA SCATTER (PNG)
    # =======================
    plt.figure(figsize=(10, 7))
    for c in sorted(df["cluster"].unique()):
        m = df["cluster"] == c
        plt.scatter(df.loc[m, "pca1"], df.loc[m, "pca2"], s=8, alpha=0.4, label=f"Cluster {c}")

    plt.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c="black", s=140, marker="x", linewidths=3, label="Centroids"
    )
    plt.title("PCA 2D Cluster Map (Main)")
    plt.xlabel("pca1")
    plt.ylabel("pca2")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =======================
    # DISTANCE HIST (PNG)
    # =======================
    intra = np.linalg.norm(X2 - centroids_2d[labels], axis=1)  # point -> own centroid
    d_all = pairwise_distances(X2, centroids_2d)              # point -> all centroids
    d_all[np.arange(len(labels)), labels] = np.inf
    inter = np.min(d_all, axis=1)                             # point -> closest other centroid

    intra_mean, inter_mean = float(np.mean(intra)), float(np.mean(inter))
    ratio = intra_mean / (inter_mean + 1e-9)

    print("\n===== DISTANCE SUMMARY (PCA 2D) =====")
    print(f"Intra mean: {intra_mean:.4f}")
    print(f"Inter mean: {inter_mean:.4f}")
    print(f"Mean ratio (intra/inter): {ratio:.4f}  (küçük olması iyi)")

    plt.figure(figsize=(9, 4.5))
    plt.hist(intra, bins=40, alpha=0.7, label="Intra (own centroid)")
    plt.hist(inter, bins=40, alpha=0.7, label="Inter (nearest other centroid)")
    plt.title("Intra vs Inter Distance Histogram (Main, PCA 2D)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "distance_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =======================
    # CLUSTER × GENRE TABLE (PNG)
    # =======================
    if "playlist_genre" in df.columns:
        counts = (
            df.groupby(["cluster", "playlist_genre"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
        pct.index = [f"Cluster {i}" for i in pct.index]
        save_table_png(pct, OUT_DIR / "cluster_genre_table.png", "Cluster × Genre Distribution (%) (Main)")
        print(f"\n📸 PNG tablo kaydedildi: {OUT_DIR / 'cluster_genre_table.png'}")
    else:
        print("\nℹ playlist_genre yok → genre tablosu üretilmedi.")

    print(f"\n MAIN çıktılar: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
