# config.py
from __future__ import annotations

# ----------------------------
# NRC Lex duyguları (10 duygu)
# ----------------------------
NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative",
]

# ----------------------------
# Spotify audio feature set (mode ve key eklendi)
# ----------------------------
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "mode",  # Major (1) / Minor (0) ayrımı için
    "key",   # Şarkının tonu için
]

# ----------------------------
# Model ayarları
# ----------------------------
PCA_VARIANCE = 0.95
BEST_K = 4
RANDOM_STATE = 42

# ----------------------------
# Dataset
# ----------------------------
DATASET_FILENAME = "spotify_songs.csv"

# ----------------------------
# App / UI
# ----------------------------
HYBRID_DIFF_TH = 10.0  # Hibrit profil eşiği (yüzde puan)

# ----------------------------
# Subclustering
# ----------------------------
SUBCLUSTER0_K = 2  # cluster==0 alt-küme sayısı

# ----------------------------
# Artifacts
# ----------------------------
ARTIFACT_PATH = "musicdna_artifacts.joblib"

# ----------------------------
# Veri temizliği
# ----------------------------
USELESS_COLS = [
    "track_album_id",
    "track_album_name",
    "playlist_id",
    "playlist_name",
    "track_album_release_date",
]

# ----------------------------
# Lyrics feature seçimi / ağırlığı
# ----------------------------
KEEP_LYRICS = [
    "lyr_joy",
    "lyr_sadness",
    "lyr_anger",
    "lyr_positive",
    "lyr_negative",
]
LYR_WEIGHT = 2.0

# ----------------------------
# Feature boosting (Ayrışmayı artırmak için çarpanlar yükseltildi)
# ----------------------------
BOOST_FACTORS = {
    "speechiness": 1.60,      # Rap/Sözel parçaları keskin ayırmak için artırıldı
    "instrumentalness": 1.50, # Enstrümantal parçaları öne çıkarmak için artırıldı
    "tempo": 1.40,            # Hızlı/Yavaş ritimleri ayrıştırmak için eklendi
    "danceability": 1.20,
    "valence": 1.15,
    "energy": 1.30,           # Enerji ayrımını keskinleştirmek için artırıldı
    "acousticness": 1.20,
    "mode": 1.50,             # Majör/Minör zıtlığını modelde baskın kılmak için eklendi
}

# ----------------------------
# Outlier kırpma (winsorization)
# ----------------------------
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99

# ----------------------------
# Elbow plot ayarları
# ----------------------------
ELBOW_K_MIN = 2
ELBOW_K_MAX = 10