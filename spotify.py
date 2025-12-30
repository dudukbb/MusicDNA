import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from nrclex import NRCLex
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# ==========================================
# 0) DOSYA YOLU (daha sağlam)
# ==========================================
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    # PyCharm console / Jupyter gibi ortamlarda
    HERE = Path.cwd()
def find_csv_path() -> Path:
    """
    Repo içinde CSV bazen datasets/ altında, bazen kökte olabiliyor.
    En olası lokasyonları sırayla dener.
    """
    candidates = [
        HERE / "datasets" / "spotify_songs.csv",
        HERE / "spotify_songs.csv",
        HERE.parent / "datasets" / "spotify_songs.csv",
        HERE.parent / "spotify_songs.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "spotify_songs.csv bulunamadı. datasets/spotify_songs.csv veya proje köküne koy."
    )

# ==========================================
# 1) HELPER’LAR
# ==========================================
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Duplicates #####################")
    print("Total Duplicates:", dataframe.duplicated().sum())
    if "track_id" in dataframe.columns:
        print("Track ID Duplicates:", dataframe["track_id"].duplicated().sum())

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, linecolor='w', cmap='RdBu')
    plt.show()

def _normalize(s: str) -> str:
    return str(s).lower().strip()

def find_song_row(df, track_name, track_artist=None):
    """
    track_name ile contains (regex=False) arar.
    track_artist verilirse sonuçları sanatçıya göre filtrelemeye çalışır.
    En popüler olanı döndürür.
    """
    if not track_name or not isinstance(track_name, str) or track_name.strip() == "":
        return None

    tn = _normalize(track_name)
    mask = df["track_name"].str.lower().str.contains(tn, na=False, regex=False)
    hits = df[mask].copy()
    if hits.empty:
        return None

    if track_artist and isinstance(track_artist, str) and track_artist.strip() != "":
        ta = _normalize(track_artist)
        artist_mask = hits["track_artist"].str.lower().str.contains(ta, na=False, regex=False)
        if artist_mask.any():
            hits = hits[artist_mask]

    return hits.sort_values("track_popularity", ascending=False).iloc[0]

# ==========================================
# 2) NRCLex
# ==========================================
NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative"
]

def nrclex_features(text: str) -> dict:
    """Lyrics metninden duygu skorları çıkarır (token sayısına göre normalize)."""
    if not isinstance(text, str) or text.strip() == "":
        return {f"lyr_{e}": 0.0 for e in NRC_EMOTIONS}
    lex = NRCLex(text)
    raw = lex.raw_emotion_scores
    token_count = len(lex.words) if hasattr(lex, "words") else 0
    denom = token_count if token_count > 0 else 1
    return {f"lyr_{e}": raw.get(e, 0) / denom for e in NRC_EMOTIONS}

# ==========================================
# 3) “FAL VARİ” YORUM SİSTEMİ (Yöntem 2)
# ==========================================
PERSONA_TEXT = {
    "Mutlu & Coşkulu": [
        "Sen müziği sadece dinlemiyorsun; enerji topluyorsun.",
        "Pozitif tınılar sende anında mod yükseltir.",
        "Neşeli parçalar seni hızla motive eden bir düğme gibi."
    ],
    "Agresif & Dinamik": [
        "Müzikte güç ve sertlik seni canlı tutuyor.",
        "Enerji yükseldikçe sen daha net hissediyorsun.",
        "Sert ritimler sende bir ‘gaz’ etkisi yaratıyor."
    ],
    "Hüzünlü & Melankolik": [
        "Müzik senin için yüzleşme alanı gibi; duygudan kaçmıyorsun.",
        "Derin ve duygusal parçalar sende daha uzun kalıyor.",
        "Herkes ritme kapılırken sen sözlere takılan taraftasın."
    ],
    "Huzurlu & Sakin": [
        "Müzik sende gürültüyü azaltan bir filtre gibi çalışıyor.",
        "Sakin tınılar senin denge alanın.",
        "Akış ve dinginlik, zevkinin gizli anahtarı."
    ],
    "Dans/Party": [
        "Ritmi yakaladığında ayakta durmak zor.",
        "Beat iyiyse gerisi teferruat.",
        "Playlist’in ‘kıpır kıpır’ çalışıyor."
    ],
    "Enstrümantal/Odak": [
        "Sözlerden çok melodilerle bağ kuruyorsun.",
        "Müzik sende bazen ‘odak modu’ açıyor.",
        "Enstrümanlar konuşsun, sen dinlersin."
    ],
    "Sözel/Rap": [
        "Sen ritim kadar mesajı da takip ediyorsun.",
        "Anlatısı olan parçalar seni daha çabuk yakalıyor.",
        "Söz ağırlığı sende bir ‘hikâye’ hissi bırakıyor."
    ],
    "Akustik": [
        "Doğal tınılar sana daha gerçek geliyor.",
        "Akustik taraf sende ‘samimiyet’ arayışı gibi.",
        "Yapay olmayan sesler sende daha çok iz bırakıyor."
    ],
}

ENDING_BY_MOOD = {
    "Mutlu & Coşkulu": [
        "Bu enerji seni kolay kolay aşağı çekmez.",
        "Pozitiflik senin temel frekansın.",
        "Mod yükselten tarafın oldukça baskın."
    ],
    "Agresif & Dinamik": [
        "Güçlü hissettiğinde müzik seninle aynı yönde akıyor.",
        "Bu sertlik sende net bir duruş yaratıyor."
    ],
    "Hüzünlü & Melankolik": [
        "Bu derinlik herkeste bulunmaz.",
        "Duygularla bu kadar yakın olmak güçlü bir taraf."
    ],
    "Huzurlu & Sakin": [
        "Bu sakinlik senin doğal alanın.",
        "Denge arayışın müzikte çok net hissediliyor."
    ],
}

GENERAL_ENDINGS = [
    "Bu müzik zevki sana özgü bir imza gibi.",
    "Müzik tercihlerin rastgele değil, oldukça tutarlı.",
    "Bu dinleme tarzı seni anlatmanın sessiz bir yolu.",
    "Playlist’inde bile bir karakter var."
]

def seed_from_user_songs(user_song_list) -> int:
    key = "|".join([f"{s.get('name','')}-{s.get('artist','')}" for s in user_song_list]).lower()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def generate_dynamic_comment(cluster_label: str, seed: int | None = None, max_sentences=4) -> str:
    """
    - Aynı DNA etiketi için farklı kullanıcı -> (genelde) farklı yorum (seed farklı)
    - Aynı kişi aynı şarkılar -> aynı yorum (seed sabit)
    """
    rng = random.Random(seed)
    parts = [p.strip() for p in cluster_label.split("/")] if cluster_label else []

    sentences = []
    for p in parts:
        if p in PERSONA_TEXT:
            sentences.append(rng.choice(PERSONA_TEXT[p]))

    rng.shuffle(sentences)
    sentences = sentences[:max(1, max_sentences - 1)]

    main_mood = parts[0] if parts else None
    if main_mood in ENDING_BY_MOOD:
        sentences.append(rng.choice(ENDING_BY_MOOD[main_mood]))
    else:
        sentences.append(rng.choice(GENERAL_ENDINGS))

    return "\n".join(sentences)

# ==========================================
# 4) VERİ OKU + PREPROCESS
# ==========================================
csv_path = find_csv_path()
df = pd.read_csv(csv_path)

# lyrics/language olmayanları at (NRCLex için)
df.dropna(subset=["lyrics", "language"], inplace=True)

# işimize yaramayacak columnları sil
useless_cols = [
    "track_album_id",
    "track_album_name",
    "playlist_id",
    "playlist_name",
    "track_album_release_date"
]
df.drop([c for c in useless_cols if c in df.columns], axis=1, inplace=True)

# Duplicate track_id temizle
if "track_id" in df.columns:
    df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)

# Audio feature set (senin datasetine göre)
audio_features = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo"
]

print("Adım 1: Lyrics duygu analizi (NRCLex) yapılıyor...")
lyrics_results = df["lyrics"].apply(nrclex_features)
lyrics_feat_df = pd.DataFrame(lyrics_results.tolist())
actual_lyrics_features = [c for c in lyrics_feat_df.columns if c.startswith("lyr_")]

df_final = pd.concat([df.reset_index(drop=True), lyrics_feat_df.reset_index(drop=True)], axis=1)

model_features = audio_features + actual_lyrics_features

# ==========================================
# 5) SCALER + PCA
# ==========================================
X = df_final[model_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Orijinal feature sayısı:", X.shape[1])
print("PCA sonrası bileşen sayısı:", X_pca.shape[1])
print("Açıklanan toplam varyans oranı:", pca.explained_variance_ratio_.sum())

# ==========================================
# 6) KMEANS (sabit k)
# ==========================================
# Elbow (opsiyonel)
ssd = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    ssd.append(km.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(list(K_range), ssd, "bx-", markersize=8, linewidth=2)
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Inertia (SSD)")
plt.title("İdeal Küme Sayısı İçin Dirsek Metodu (Elbow)")
plt.grid(True)
plt.show()

best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_final["cluster"] = kmeans.fit_predict(X_pca)

print(df_final["cluster"].value_counts())

# ==========================================
# 7) CLUSTER İSİMLENDİRME (max 3 tag + top-k)
# ==========================================
cluster_profile = df_final.groupby("cluster")[model_features].mean()
cluster_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
cluster_z = cluster_z.replace([np.inf, -np.inf], np.nan).fillna(0)

def name_cluster(z, max_tags=3, top_k=2, min_z=0.8):
    tags = []
    energy = float(z.get("energy", 0.0))
    valence = float(z.get("valence", 0.0))

    # Mood
    if energy > 0 and valence > 0:
        tags.append("Mutlu & Coşkulu")
    elif energy > 0 and valence <= 0:
        tags.append("Agresif & Dinamik")
    elif energy <= 0 and valence <= 0:
        tags.append("Hüzünlü & Melankolik")
    else:
        tags.append("Huzurlu & Sakin")

    if len(tags) >= max_tags:
        return tags[:max_tags]

    candidates = [
        ("danceability", "Dans/Party"),
        ("instrumentalness", "Enstrümantal/Odak"),
        ("speechiness", "Sözel/Rap"),
        ("acousticness", "Akustik"),
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

cluster_names = (
    cluster_z.apply(lambda row: " / ".join(name_cluster(row.to_dict())), axis=1).to_dict()
)
df_final["cluster_name"] = df_final["cluster"].map(cluster_names)

print("\n===== MÜZİKAL KİŞİLİK DAĞILIMI =====")
print(df_final["cluster_name"].value_counts())

# ==========================================
# 8) PCA 2D GÖRSEL
# ==========================================
pca_viz = PCA(n_components=2, random_state=42)
X_viz = pca_viz.fit_transform(X_pca)

df_final["pca1"], df_final["pca2"] = X_viz[:, 0], X_viz[:, 1]
centroids_2d = pca_viz.transform(kmeans.cluster_centers_)

COLOR_MAP = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
}

plt.figure(figsize=(13, 8))
clusters = sorted(df_final["cluster"].unique())

for cid in clusters:
    sub = df_final[df_final["cluster"] == cid]
    plt.scatter(
        sub["pca1"], sub["pca2"],
        s=14, alpha=0.35,
        color=COLOR_MAP.get(cid, "#333333"),
        edgecolors="none",
        label=f"{cid} - {cluster_names[cid]}"
    )

plt.scatter(
    centroids_2d[:, 0], centroids_2d[:, 1],
    marker="X", s=320,
    color="black", edgecolor="white", linewidth=1.5,
    label="Centroid"
)

for i, (x, y) in enumerate(centroids_2d):
    plt.text(
        x + 0.15, y + 0.15, f"{i}",
        fontsize=11, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.25")
    )

plt.title("MusicDNA Kümeleme Haritası (PCA 2D)", fontsize=14, fontweight="bold")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.xlim(df_final["pca1"].quantile(0.01), df_final["pca1"].quantile(0.99))
plt.ylim(df_final["pca2"].quantile(0.01), df_final["pca2"].quantile(0.99))
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10)
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()

# ==========================================
# 9) KULLANICI -> DNA + ÖNERİ
# ==========================================
def music_dna_engine(user_song_list, df=df_final, top_n=5):
    found_rows = []
    for s in user_song_list:
        row = find_song_row(df, s.get("name"), s.get("artist"))
        if row is not None:
            found_rows.append(row)

    if not found_rows:
        return None, None

    user_vec = pd.DataFrame(found_rows)[model_features].mean().to_frame().T
    user_scaled = scaler.transform(user_vec)
    user_pca = pca.transform(user_scaled)
    cluster_id = int(kmeans.predict(user_pca)[0])

    dna_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    used_names = {r["track_name"].lower() for r in found_rows if isinstance(r.get("track_name"), str)}
    recs = df[df["cluster"] == cluster_id].copy()
    recs = recs[~recs["track_name"].str.lower().isin(used_names)]
    top_recs = recs.sort_values("track_popularity", ascending=False).head(top_n)

    return dna_name, top_recs[["track_name", "track_artist", "track_popularity"]]

# ==========================================
# 10) ARTIFACT EXPORT
# ==========================================
from export_artifacts import export_artifacts

export_artifacts(
    df_final=df_final,
    scaler=scaler,
    pca=pca,
    kmeans=kmeans,
    model_features=model_features,
    cluster_names=cluster_names,
    out_path="musicdna_artifacts.joblib",
)

# ==========================================
# 11) DEMO
# ==========================================
demo_songs = [{"name": "Anaconda", "artist": "Nicki Minaj"}, {"name": "Shape of You"}]
dna, recs = music_dna_engine(demo_songs)

print("\n===== DEMO SONUÇ =====")
if dna is None:
    print("Şarkı bulunamadı.")
else:
    seed = seed_from_user_songs(demo_songs)
    print(f"MusicDNA Profilin: {dna}\n")
    print("🔮 Kişilik Yorumu:")
    print(generate_dynamic_comment(dna, seed=seed))
    print("\nDNA'na Uygun Tavsiyeler:")
    print(recs)

