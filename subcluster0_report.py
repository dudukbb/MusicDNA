from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import pairwise_distances

from artifacts import load_artifacts
from config import ARTIFACT_PATH, RANDOM_STATE, SUBCLUSTER0_K


OUT_DIR = Path("outputs/subcluster0")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_table_png(pivot_pct: pd.DataFrame, out_path: Path, title: str) -> None:
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

    if not {"pca1", "pca2", "cluster"}.issubset(df.columns):
        raise ValueError("df_final içinde pca1/pca2/cluster yok. Önce `python pipeline.py` çalıştır.")

    df0 = df[df["cluster"] == 0].copy()
    if df0.empty:
        raise ValueError("Cluster 0 boş görünüyor. df_final['cluster'] değerlerini kontrol et.")

    X2 = df0[["pca1", "pca2"]].to_numpy(float)

    k_sub = int(SUBCLUSTER0_K) if "SUBCLUSTER0_K" in dir() else 2
    km = KMeans(n_clusters=k_sub, random_state=RANDOM_STATE, n_init=10)
    sub_labels = km.fit_predict(X2)
    centroids_2d = km.cluster_centers_

    # metrik
    sil = float(silhouette_score(X2, sub_labels))

    print("\n===== SUBCLUSTER 0 REPORT (PCA 2D) =====")
    print(f"Sub-K: {len(np.unique(sub_labels))}")
    print(f"Silhouette (subcluster0): {sil:.4f}")

    # =======================
    # SILHOUETTE HIST (PNG)
    # =======================
    sil_vals = silhouette_samples(X2, sub_labels)
    plt.figure(figsize=(9, 4.5))
    plt.hist(sil_vals, bins=40)
    plt.title("Silhouette Distribution (Subcluster 0, PCA 2D)")
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
    for c in sorted(np.unique(sub_labels)):
        m = sub_labels == c
        plt.scatter(X2[m, 0], X2[m, 1], s=10, alpha=0.5, label=f"Subcluster {c}")

    plt.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c="black", s=140, marker="x", linewidths=3, label="Centroids"
    )
    plt.title("PCA 2D – Subclusters inside Cluster 0")
    plt.xlabel("pca1")
    plt.ylabel("pca2")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_subclusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =======================
    # DISTANCE HIST (PNG)
    # =======================
    intra = np.linalg.norm(X2 - centroids_2d[sub_labels], axis=1)
    d_all = pairwise_distances(X2, centroids_2d)
    d_all[np.arange(len(sub_labels)), sub_labels] = np.inf
    inter = np.min(d_all, axis=1)

    intra_mean, inter_mean = float(np.mean(intra)), float(np.mean(inter))
    ratio = intra_mean / (inter_mean + 1e-9)

    print("\n===== DISTANCE SUMMARY (Subcluster 0, PCA 2D) =====")
    print(f"Intra mean: {intra_mean:.4f}")
    print(f"Inter mean: {inter_mean:.4f}")
    print(f"Mean ratio (intra/inter): {ratio:.4f}  (küçük olması iyi)")

    plt.figure(figsize=(9, 4.5))
    plt.hist(intra, bins=40, alpha=0.7, label="Intra (own centroid)")
    plt.hist(inter, bins=40, alpha=0.7, label="Inter (nearest other centroid)")
    plt.title("Intra vs Inter Distance Histogram (Subcluster 0, PCA 2D)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "distance_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =======================
    # SUBCLUSTER × GENRE TABLE (PNG)
    # =======================
    if "playlist_genre" in df0.columns:
        df0["subcluster0"] = sub_labels

        counts = (
            df0.groupby(["subcluster0", "playlist_genre"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
        pct.index = [f"Subcluster {i}" for i in pct.index]

        save_table_png(pct, OUT_DIR / "subcluster_genre_table.png",
                       "Subcluster(Cluster 0) × Genre Distribution (%)")
        print(f"\n📸 PNG tablo kaydedildi: {OUT_DIR / 'subcluster_genre_table.png'}")
    else:
        print("\nℹ playlist_genre yok → subcluster genre tablosu üretilmedi.")

    print(f"\n SUBCLUSTER0 çıktılar: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
