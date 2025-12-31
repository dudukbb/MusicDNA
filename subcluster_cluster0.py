# subcluster_cluster0.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import RANDOM_STATE, SUBCLUSTER0_K
from metrics_utils import compute_silhouette, intra_cluster_distance, inter_centroid_distance


def _get_pca_cols(df: pd.DataFrame) -> list[str]:
    """Varsa pca1..pca5'i kullan."""
    return [c for c in ["pca1", "pca2", "pca3", "pca4", "pca5"] if c in df.columns]


def split_cluster0_pop_rap(
    df_final: pd.DataFrame,
    sub_k: int = SUBCLUSTER0_K,
    pca_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Sadece cluster==0 için alt-kümeleme yapar (Pop vs Rap).
    Ana cluster (BEST_K) değişmez.
    """
    if sub_k != 2:
        raise ValueError("Bu sürüm Pop vs Rap için tasarlandı: sub_k=2 olmalı.")

    if pca_cols is None:
        pca_cols = _get_pca_cols(df_final)

    sub_mask = df_final["cluster"] == 0
    sub = df_final.loc[sub_mask].copy()

    if sub.empty:
        print("Cluster 0 bulunamadı. (cluster==0 yok)")
        df_final = df_final.copy()
        df_final["cluster0_sub"] = np.nan
        df_final["cluster0_sub_name"] = np.nan
        return df_final

    if not pca_cols:
        raise ValueError("PCA kolonları bulunamadı (pca1..pca5). Önce pipeline.py çalışmalı.")

    X_sub = sub[pca_cols].values

    km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    sub_labels = km.fit_predict(X_sub)
    sub["cluster0_sub"] = sub_labels

    # İsimlendirme: speechiness düşük -> Pop, yüksek -> Rap
    if "speechiness" in sub.columns:
        means = sub.groupby("cluster0_sub")["speechiness"].mean().sort_values()
        pop_id = int(means.index[0])
        rap_id = int(means.index[-1])
    else:
        # speechiness yoksa fallback
        pop_id, rap_id = 0, 1

    name_map = {pop_id: "Pop", rap_id: "Rap"}
    sub["cluster0_sub_name"] = sub["cluster0_sub"].map(name_map)

    df_out = df_final.copy()
    df_out["cluster0_sub"] = np.nan
    df_out["cluster0_sub_name"] = np.nan
    df_out.loc[sub_mask, "cluster0_sub"] = sub["cluster0_sub"].values
    df_out.loc[sub_mask, "cluster0_sub_name"] = sub["cluster0_sub_name"].values
    return df_out


def elbow_cluster0(df_final: pd.DataFrame, k_range=range(2, 7), pca_cols=None):
    if pca_cols is None:
        pca_cols = _get_pca_cols(df_final)

    sub = df_final[df_final["cluster"] == 0].copy()
    if sub.empty:
        print("Cluster 0 yok, elbow çizilemiyor.")
        return

    X = sub[pca_cols].values
    ks = list(k_range)
    ssd = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        ssd.append(km.inertia_)

    plt.figure(figsize=(8, 4.5))
    plt.plot(ks, ssd, marker="o")
    plt.xticks(ks)
    plt.xlabel("k (cluster0 sub)")
    plt.ylabel("Inertia (SSD)")
    plt.title("Elbow Plot (Cluster 0 Subclustering)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cluster0_split(df_final: pd.DataFrame):
    """PCA-1 vs PCA-2: cluster0 içindeki Pop/Rap."""
    c0 = df_final[df_final["cluster"] == 0].copy()
    if c0.empty:
        print("Cluster 0 yok, plot çizilemiyor.")
        return

    plt.figure(figsize=(10, 7))
    pop = c0[c0["cluster0_sub_name"] == "Pop"]
    rap = c0[c0["cluster0_sub_name"] == "Rap"]

    plt.scatter(pop["pca1"], pop["pca2"], s=14, alpha=0.75, marker="o", label="Cluster 0 - Pop")
    plt.scatter(rap["pca1"], rap["pca2"], s=14, alpha=0.75, marker="^", label="Cluster 0 - Rap")

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Cluster 0 Split (Pop vs Rap) - PCA 2D")
    plt.legend(markerscale=1.5, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_as_groups(df_final: pd.DataFrame):
    """
    Tüm grupları birlikte göster:
    - cluster0-pop
    - cluster0-rap
    - cluster1..(BEST_K-1)
    """
    plt.figure(figsize=(10, 7))

    # diğer clusterlar
    for cid in sorted([c for c in df_final["cluster"].unique() if c != 0]):
        sub = df_final[df_final["cluster"] == cid]
        plt.scatter(sub["pca1"], sub["pca2"], s=8, alpha=0.25, label=f"Cluster {cid}")

    # cluster0 pop/rap
    c0 = df_final[df_final["cluster"] == 0]
    pop = c0[c0["cluster0_sub_name"] == "Pop"]
    rap = c0[c0["cluster0_sub_name"] == "Rap"]

    plt.scatter(pop["pca1"], pop["pca2"], s=14, alpha=0.75, marker="o", label="Cluster 0 - Pop")
    plt.scatter(rap["pca1"], rap["pca2"], s=14, alpha=0.75, marker="^", label="Cluster 0 - Rap")

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("All Groups (Cluster0 split k=2) - PCA 2D")
    plt.legend(markerscale=1.5, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def recommend_cases(df_final: pd.DataFrame, n_each: int = 5):
    """Pop ve Rap alt-kümelerinden örnek case listesi basar."""
    cols = [c for c in ["track_name", "track_artist", "artist_name", "track_id"] if c in df_final.columns]
    show_cols = ["cluster", "cluster0_sub_name"] + cols

    c0 = df_final[df_final["cluster"] == 0].copy()
    pop_sub = c0[c0["cluster0_sub_name"] == "Pop"]
    rap_sub = c0[c0["cluster0_sub_name"] == "Rap"]

    pop = pop_sub.sample(min(n_each, len(pop_sub)), random_state=RANDOM_STATE) if not pop_sub.empty else pop_sub
    rap = rap_sub.sample(min(n_each, len(rap_sub)), random_state=RANDOM_STATE) if not rap_sub.empty else rap_sub

    print("\n=== POP case önerileri ===")
    print(pop[show_cols].head(n_each).to_string(index=False))

    print("\n=== RAP case önerileri ===")
    print(rap[show_cols].head(n_each).to_string(index=False))


def main():
    from pipeline import build_artifacts  # circular riskini azaltır

    df_final, scaler, pca, kmeans, model_features, cluster_names = build_artifacts()

    # 1) cluster0 elbow + split
    elbow_cluster0(df_final, k_range=range(2, 7))
    df_final2 = split_cluster0_pop_rap(df_final, sub_k=2)

    # 2) cluster0 split PCA 2D
    plot_cluster0_split(df_final2)

    # 3) tüm gruplar birlikte
    plot_all_as_groups(df_final2)

    # 4) metrikler (cluster0 split için)
    c0 = df_final2[df_final2["cluster"] == 0].copy()
    y = c0["cluster0_sub"].astype(int).values
    pca_cols = _get_pca_cols(c0)
    X = c0[pca_cols].values

    sil = compute_silhouette(X, y)
    intra = intra_cluster_distance(X, y)
    inter = inter_centroid_distance(X, y)
    sep_ratio = (inter / intra) if intra > 0 else float("nan")

    print("\n===== METRICS (CLUSTER0 SPLIT k=2) =====")
    print(f"Silhouette (cluster0 split): {sil:.3f}")
    print(f"Intra distance (avg): {intra:.3f}")
    print(f"Inter centroid distance (avg): {inter:.3f}")
    print(f"Separation ratio (inter/intra): {sep_ratio:.3f}")

    # 5) örnek case önerileri
    recommend_cases(df_final2, n_each=5)


if __name__ == "__main__":
    main()
