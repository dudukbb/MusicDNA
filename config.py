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
# Spotify audio feature set
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
# Feature boosting (opsiyonel ağırlıklandırma)
# ----------------------------
BOOST_FACTORS = {
    "speechiness": 1.30,
    "instrumentalness": 1.25,
    "danceability": 1.20,
    "valence": 1.15,
    "energy": 1.10,
    "acousticness": 1.10,
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
