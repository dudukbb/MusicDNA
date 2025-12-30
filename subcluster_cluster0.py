# subcluster_cluster0.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import RANDOM_STATE
from pipeline import build_artifacts
from metrics_utils import compute_silhouette, intra_cluster_distance, inter_centroid_distance


def split_cluster0_pop_rap(
    df_final: pd.DataFrame,
    sub_k: int = 2,
    pca_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Sadece cluster==0 için alt-kümeleme yapar (Pop vs Rap).
    Ana cluster (BEST_K) değişmez.
    """
    if pca_cols is None:
        pca_cols = [c for c in ["pca1", "pca2", "pca3", "pca4", "pca5"] if c in df_final.columns]

    sub_mask = df_final["cluster"] == 0
    sub = df_final.loc[sub_mask].copy()

    if sub.empty:
        print("Cluster 0 bulunamadı. (cluster==0 yok)")
        df_final["cluster0_sub"] = np.nan
        df_final["cluster0_sub_name"] = np.nan
        return df_final

    X_sub = sub[pca_cols].values
    km2 = KMeans(n_clusters=sub_k, random_state=RANDOM_STATE, n_init=10)
    sub_labels = km2.fit_predict(X_sub)
    sub["cluster0_sub"] = sub_labels

    # Pop vs Rap etiketi: speechiness yüksek olanı "Rap"
    if "speechiness" in sub.columns:
        means = sub.groupby("cluster0_sub")["speechiness"].mean().sort_values()
        pop_id = int(means.index[0])
        rap_id = int(means.index[-1])
    else:
        pop_id, rap_id = 0, 1

    name_map = {pop_id: "Pop", rap_id: "Rap"}
    sub["cluster0_sub_name"] = sub["cluster0_sub"].map(name_map)

    df_final = df_final.copy()
    df_final["cluster0_sub"] = np.nan
    df_final["cluster0_sub_name"] = np.nan
    df_final.loc[sub_mask, "cluster0_sub"] = sub["cluster0_sub"].values
    df_final.loc[sub_mask, "cluster0_sub_name"] = sub["cluster0_sub_name"].values

    return df_final


def elbow_cluster0(df_final: pd.DataFrame, k_range=range(2, 7), pca_cols=None):
    if pca_cols is None:
        pca_cols = [c for c in ["pca1", "pca2", "pca3", "pca4", "pca5"] if c in df_final.columns]

    sub = df_final[df_final["cluster"] == 0].copy()
    X = sub[pca_cols].values

    ssd = []
    ks = list(k_range)
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
    """
    PCA-1 vs PCA-2 plot:
    - cluster 0 içindeki Pop/Rap farklı marker ile gösterilir
    """
    plt.figure(figsize=(10, 7))

    c0 = df_final[df_final["cluster"] == 0]
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


def plot_all_as_5_groups(df_final: pd.DataFrame):
    """
    5 grup görünsün:
    - cluster0-pop
    - cluster0-rap
    - cluster1
    - cluster2
    - cluster3
    """
    plt.figure(figsize=(10, 7))

    # cluster 1-3
    for cid in sorted([c for c in df_final["cluster"].unique() if c != 0]):
        sub = df_final[df_final["cluster"] == cid]
        plt.scatter(sub["pca1"], sub["pca2"], s=8, alpha=0.25, label=f"Cluster {cid}")

    # cluster 0 pop/rap
    c0 = df_final[df_final["cluster"] == 0]
    pop = c0[c0["cluster0_sub_name"] == "Pop"]
    rap = c0[c0["cluster0_sub_name"] == "Rap"]
    plt.scatter(pop["pca1"], pop["pca2"], s=14, alpha=0.75, marker="o", label="Cluster 0 - Pop")
    plt.scatter(rap["pca1"], rap["pca2"], s=14, alpha=0.75, marker="^", label="Cluster 0 - Rap")

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("All Groups (Cluster0 split) - PCA 2D")
    plt.legend(markerscale=1.5, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def recommend_cases(df_final: pd.DataFrame, n_each: int = 5):
    """
    Pop ve Rap alt-kümesinden örnek şarkı case listesi basar.
    (track_name/artist_name kolonları varsa onları kullanır)
    """
    cols = []
    for c in ["track_name", "track_artist", "artist_name", "track_id"]:
        if c in df_final.columns:
            cols.append(c)

    base_cols = ["cluster", "cluster0_sub_name"]
    show_cols = base_cols + cols if cols else base_cols

    c0 = df_final[df_final["cluster"] == 0].copy()
    pop = c0[c0["cluster0_sub_name"] == "Pop"].sample(min(n_each, len(c0)), random_state=RANDOM_STATE)
    rap = c0[c0["cluster0_sub_name"] == "Rap"].sample(min(n_each, len(c0)), random_state=RANDOM_STATE)

    print("\n=== POP case önerileri ===")
    print(pop[show_cols].head(n_each).to_string(index=False))

    print("\n=== RAP case önerileri ===")
    print(rap[show_cols].head(n_each).to_string(index=False))


def main():
    df_final, scaler, pca, kmeans, model_features, cluster_names = build_artifacts()

    # 1) cluster0 elbow + split
    elbow_cluster0(df_final, k_range=range(2, 7))
    df_final2 = split_cluster0_pop_rap(df_final, sub_k=2)

    # 2) cluster0 split PCA 2D
    plot_cluster0_split(df_final2)

    # 3) 5 grup beraber görünsün
    plot_all_as_5_groups(df_final2)

    # 4) metrikler: cluster0 split için
    c0 = df_final2[df_final2["cluster"] == 0].copy()
    # Pop/Rap label -> 0/1 label
    y = (c0["cluster0_sub_name"] == "Rap").astype(int).values
    X = c0[[c for c in ["pca1", "pca2", "pca3", "pca4", "pca5"] if c in c0.columns]].values

    sil = compute_silhouette(X, y)
    intra = intra_cluster_distance(X, y)
    inter = inter_centroid_distance(X, y)
    sep_ratio = (inter / intra) if intra > 0 else float("nan")

    print("\n===== METRICS (CLUSTER0 SPLIT) =====")
    print(f"Silhouette (cluster0 split): {sil:.3f}")
    print(f"Intra distance (avg): {intra:.3f}")
    print(f"Inter centroid distance (avg): {inter:.3f}")
    print(f"Separation ratio (inter/intra): {sep_ratio:.3f}")

    # 5) case önerileri
    recommend_cases(df_final2, n_each=5)


if __name__ == "__main__":
    main()
