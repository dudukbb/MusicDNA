# pipeline.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nrclex import NRCLex

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from config import (
    NRC_EMOTIONS,
    AUDIO_FEATURES,
    PCA_VARIANCE,
    BEST_K,
    RANDOM_STATE,
    USELESS_COLS,
    ARTIFACT_PATH,
    KEEP_LYRICS,
    LYR_WEIGHT,
    BOOST_FACTORS,
    WINSOR_Q_LOW,
    WINSOR_Q_HIGH,
    ELBOW_K_MIN,
    ELBOW_K_MAX,
)
from utils import find_csv_path
from artifacts import export_artifacts
from metrics_utils import (
    plot_corr_heatmap,
    compute_silhouette,
    intra_cluster_distance,
    inter_centroid_distance,
)


def nrclex_features(text: str) -> dict:
    """Lyrics metninden duygu skorları çıkarır (token sayısına göre normalize)."""
    if not isinstance(text, str) or text.strip() == "":
        return {f"lyr_{e}": 0.0 for e in NRC_EMOTIONS}

    lex = NRCLex(text)
    raw = lex.raw_emotion_scores
    token_count = len(lex.words) if hasattr(lex, "words") else 0
    denom = token_count if token_count > 0 else 1
    return {f"lyr_{e}": raw.get(e, 0) / denom for e in NRC_EMOTIONS}


def name_cluster(z: dict, max_tags: int = 3, top_k: int = 2, min_z: float = 0.8):
    """
    Cluster isimlendirme:
    1) Ana etiket: Energy + Valence ile mood bölgesi
    2) Secondary etiketler: dans / söz / akustik / enstrümantal
    """
    energy = float(z.get("energy", 0.0))
    valence = float(z.get("valence", 0.0))

    th = 0.2
    tags: list[str] = []
    mood_label = "Medium Energy – Balanced Valence"

    if energy > th and valence > th:
        tags.append("Coşkulu & Dans")
        mood_label = "High Energy – Positive Valence"
    elif energy > th and valence < -th:
        tags.append("Sert & Yoğun")
        mood_label = "High Energy – Negative Valence"
    elif energy < -th and valence < -th:
        tags.append("Melankolik & Düşük Enerji")
        mood_label = "Low Energy – Negative Valence"
    elif energy < -th and valence > th:
        tags.append("Sakin & Huzurlu")
        mood_label = "Low Energy – Positive Valence"
    else:
        tags.append("Dengeli & Hibrit")
        mood_label = "Medium Energy – Balanced Valence"

    candidates = [
        ("danceability", "Dans"),
        ("speechiness", "Sözel"),
        ("acousticness", "Akustik"),
        ("instrumentalness", "Enstrümantal"),
    ]

    scored = [(float(z.get(k, 0.0)), label) for k, label in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    remaining = max_tags - len(tags)
    take = min(top_k, remaining)

    picked = 0
    for score, label in scored:
        if picked >= take:
            break
        if score >= min_z:
            tags.append(label)
            picked += 1

    if picked == 0 and remaining > 0:
        tags.append(scored[0][1])

    return tags[:max_tags], mood_label


def build_artifacts(csv_path: str | None = None):
    # 1) Load
    if csv_path is None:
        csv_path = str(find_csv_path())
    df = pd.read_csv(csv_path)

    # 2) Basic clean
    df.dropna(subset=["lyrics", "language"], inplace=True)
    df.drop([c for c in USELESS_COLS if c in df.columns], axis=1, inplace=True)

    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)

    # 3) NRC Lex features
    print("Duygu analizi yapılıyor...")
    lyrics_feat_df = pd.DataFrame(df["lyrics"].apply(nrclex_features).tolist())

    lyric_cols_all = [c for c in lyrics_feat_df.columns if c.startswith("lyr_")]
    lyric_cols = [c for c in lyric_cols_all if c in KEEP_LYRICS]
    lyrics_feat_df = lyrics_feat_df[lyric_cols]

    df_final = pd.concat(
        [df.reset_index(drop=True), lyrics_feat_df.reset_index(drop=True)],
        axis=1,
    )

    model_features = AUDIO_FEATURES + lyric_cols

    # 4) Feature set
    X = df_final[model_features].copy()

    # (A) Lyrics ağırlığı
    if lyric_cols and float(LYR_WEIGHT) != 1.0:
        X[lyric_cols] = X[lyric_cols] * float(LYR_WEIGHT)

    # (B) Feature boosting
    for col, factor in BOOST_FACTORS.items():
        if col in X.columns:
            X[col] = X[col] * float(factor)

    # (C) Winsorization
    q_low = float(WINSOR_Q_LOW)
    q_high = float(WINSOR_Q_HIGH)
    for col in X.columns:
        low = X[col].quantile(q_low)
        high = X[col].quantile(q_high)
        X[col] = X[col].clip(lower=low, upper=high)

    # Korelasyon heatmap (opsiyonel)
    print("\n[Korelasyon] Model feature'ları (audio + lyrics) korelasyon matrisi çiziliyor...")
    _ = plot_corr_heatmap(df_final, model_features, "Correlation (Audio + Lyrics)")

    audio_only = [c for c in AUDIO_FEATURES if c in df_final.columns]
    print("\n[Korelasyon] Sadece audio feature'lar korelasyon matrisi çiziliyor...")
    _ = plot_corr_heatmap(df_final, audio_only, "Correlation (Audio Only)")

    # Scale + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    print("Orijinal feature sayısı:", X.shape[1])
    print("PCA sonrası bileşen sayısı:", X_pca.shape[1])
    print("Açıklanan toplam varyans oranı:", pca.explained_variance_ratio_.sum())

    # PCA kolonları (alt kümeleme için: pca1..pca5)
    n_keep = min(5, X_pca.shape[1])
    for i in range(n_keep):
        df_final[f"pca{i+1}"] = X_pca[:, i]

    # Elbow (main)
    ssd = []
    K_range = range(int(ELBOW_K_MIN), int(ELBOW_K_MAX) + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_pca)
        ssd.append(km.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(list(K_range), ssd, marker="o")
    plt.xticks(list(K_range))
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (SSD)")
    plt.title("Elbow Plot (Main Clustering, PCA space)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5) KMeans (ANA KÜMELEME)
    kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10)
    df_final["cluster"] = kmeans.fit_predict(X_pca)

    # Metrikler
    sil = compute_silhouette(X_pca, df_final["cluster"].values)
    intra = intra_cluster_distance(X_pca, df_final["cluster"].values)
    inter = inter_centroid_distance(X_pca, df_final["cluster"].values)
    sep_ratio = (inter / intra) if intra > 0 else float("nan")

    print("\n===== METRICS (MAIN) =====")
    print(f"Silhouette (main): {sil:.3f}")
    print(f"Intra-cluster distance (avg): {intra:.3f}")
    print(f"Inter-centroid distance (avg): {inter:.3f}")
    print(f"Separation ratio (inter/intra): {sep_ratio:.3f}")

    # 6) Cluster naming (z-score)
    cluster_profile = df_final.groupby("cluster")[model_features].mean()
    cluster_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
    cluster_z = cluster_z.replace([np.inf, -np.inf], np.nan).fillna(0)

    if "playlist_genre" in df_final.columns:
        genre_summary = (
            df_final
            .groupby(["cluster", "playlist_genre"])
            .size()
            .reset_index(name="count")
        )
        genre_summary["ratio"] = (
            genre_summary["count"]
            / genre_summary.groupby("cluster")["count"].transform("sum")
        )
    else:
        genre_summary = None

    def top_genres_for_cluster(cluster_id, genre_df, top_n=2, min_ratio=0.15):
        if genre_df is None:
            return []
        sub = genre_df[genre_df["cluster"] == cluster_id]
        sub = sub[sub["ratio"] >= min_ratio].sort_values("ratio", ascending=False)
        return sub["playlist_genre"].head(top_n).tolist()

    cluster_names = {}
    for cid, row in cluster_z.iterrows():
        tags, mood_label = name_cluster(row.to_dict())
        base_title = " & ".join(tags[:2]) if len(tags) >= 2 else tags[0]

        genres = top_genres_for_cluster(cid, genre_summary, top_n=2, min_ratio=0.15)
        if genres:
            genre_tag = "–".join([g.capitalize() for g in genres])
            title = f"{base_title} ({genre_tag})"
        else:
            title = base_title

        cluster_names[cid] = f"{title}\n{mood_label}"

    df_final["cluster_name"] = df_final["cluster"].map(cluster_names)

    print("\n===== MÜZİKAL KİŞİLİK DAĞILIMI =====")
    print(df_final["cluster_name"].value_counts())

    # PCA 2D plot
    plt.figure(figsize=(10, 7))
    for c in range(BEST_K):
        mask = df_final["cluster"] == c
        cname = cluster_names.get(c, f"Cluster {c}")
        plt.scatter(
            df_final.loc[mask, "pca1"],
            df_final.loc[mask, "pca2"],
            s=8,
            alpha=0.4,
            label=f"Cluster {c}: {cname}",
        )

    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c="black", s=120, marker="x", linewidths=3, label="Centroid",
    )

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("MusicDNA Clusters (PCA 2D)")
    plt.legend(markerscale=2, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_final, scaler, pca, kmeans, model_features, cluster_names


def main():
    df_final, scaler, pca, kmeans, model_features, cluster_names = build_artifacts()

    export_artifacts(
        df_final=df_final,
        scaler=scaler,
        pca=pca,
        kmeans=kmeans,
        model_features=model_features,
        cluster_names=cluster_names,
        out_path=ARTIFACT_PATH,
    )


if __name__ == "__main__":
    main()
