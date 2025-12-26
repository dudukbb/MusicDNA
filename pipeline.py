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
    NRC_EMOTIONS, AUDIO_FEATURES,
    PCA_VARIANCE, BEST_K, RANDOM_STATE,
    USELESS_COLS, ARTIFACT_PATH
)
from utils import find_csv_path
from artifacts import export_artifacts


def nrclex_features(text: str) -> dict:
    """Lyrics metninden duygu skorları çıkarır (token sayısına göre normalize)."""
    if not isinstance(text, str) or text.strip() == "":
        return {f"lyr_{e}": 0.0 for e in NRC_EMOTIONS}

    lex = NRCLex(text)
    raw = lex.raw_emotion_scores
    token_count = len(lex.words) if hasattr(lex, "words") else 0
    denom = token_count if token_count > 0 else 1
    return {f"lyr_{e}": raw.get(e, 0) / denom for e in NRC_EMOTIONS}


def name_cluster(z: dict, max_tags: int = 3, top_k: int = 2, min_z: float = 0.8) -> list[str]:
    """
    Cluster isimlendirme:
    1) Ana etiket: Energy + Valence ile (daha stabil) mood bölgesi
    2) İkinci etiketler: dans / söz / akustik / enstrümantal gibi davranış özellikleri
    - z değerleri cluster_z (z-score) olduğu için küçük bir dead-zone (TH) kullanıyoruz.
    """
    energy = float(z.get("energy", 0.0))
    valence = float(z.get("valence", 0.0))

    TH = 0.2  
    tags: list[str] = []

    # 1) Ana mood etiketi
    if energy > TH and valence > TH:
        tags.append("Yüksek Enerji & Pozitif (Coşkulu)")
    elif energy > TH and valence < -TH:
        tags.append("Yüksek Enerji & Negatif (Sert/Yoğun)")
    elif energy < -TH and valence < -TH:
        tags.append("Düşük Enerji & Negatif (Melankolik)")
    elif energy < -TH and valence > TH:
        tags.append("Düşük Enerji & Pozitif (Sakin)")
    else:
        tags.append("Orta Enerji & Dengeli (Hibrit)")

    if len(tags) >= max_tags:
        return tags[:max_tags]

    # 2) Secondary tags
    candidates = [
        ("danceability", "Dans Odaklı"),
        ("speechiness", "Sözel Odaklı"),
        ("acousticness", "Akustik Ağırlıklı"),
        ("instrumentalness", "Enstrümantal Odaklı"),
    ]

    scored = [(float(z.get(key, 0.0)), label) for key, label in candidates]
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

    return tags[:max_tags]


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
    print("Adım 1: Lyrics duygu analizi (NRCLex) yapılıyor...")
    lyrics_feat_df = pd.DataFrame(df["lyrics"].apply(nrclex_features).tolist())
    lyric_cols = [c for c in lyrics_feat_df.columns if c.startswith("lyr_")]

    df_final = pd.concat([df.reset_index(drop=True), lyrics_feat_df.reset_index(drop=True)], axis=1)
    model_features = AUDIO_FEATURES + lyric_cols

    # 4) Scale + PCA
    X = df_final[model_features].copy()

    # Lyrics ağırlığı 
    LYR_WEIGHT = 1.5
    if lyric_cols:
        X[lyric_cols] = X[lyric_cols] * LYR_WEIGHT

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    print("Orijinal feature sayısı:", X.shape[1])
    print("PCA sonrası bileşen sayısı:", X_pca.shape[1])
    print("Açıklanan toplam varyans oranı:", pca.explained_variance_ratio_.sum())

    # 4.5) ELBOW GRAFİĞİ (EKRANA BASAR)
    ssd = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_pca)
        ssd.append(km.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(list(K_range), ssd, marker="o")
    plt.xticks(list(K_range))
    plt.xlabel("Küme Sayısı (k)")
    plt.ylabel("Inertia (SSD)")
    plt.title("Elbow Grafiği (PCA Uzayında)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5) KMeans (config.py -> BEST_K)
    kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10)
    df_final["cluster"] = kmeans.fit_predict(X_pca)

    # 5.5) PCA 2D + KMeans görselleştirme
    plt.figure(figsize=(10, 7))
    for c in range(BEST_K):
        mask = df_final["cluster"] == c
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=8,
            alpha=0.4,
            label=f"Küme {c}"
        )

    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=120,
        marker="x",
        linewidths=3,
        label="Centroid"
    )

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("MusicDNA Kümeleme Haritası (PCA 2D)")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 6) Cluster naming (z-score)
    cluster_profile = df_final.groupby("cluster")[model_features].mean()
    cluster_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
    cluster_z = cluster_z.replace([np.inf, -np.inf], np.nan).fillna(0)

    cluster_names = cluster_z.apply(lambda row: " / ".join(name_cluster(row.to_dict())), axis=1).to_dict()
    df_final["cluster_name"] = df_final["cluster"].map(cluster_names)

    print("\n===== MÜZİKAL KİŞİLİK DAĞILIMI =====")
    print(df_final["cluster_name"].value_counts())

    # 7) PCA 2D değerlerini df'e ekle (app/analiz için)
    df_final["pca1"] = X_pca[:, 0]
    df_final["pca2"] = X_pca[:, 1]

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
