# config.py
from __future__ import annotations

# NRC Lex duyguları (eski halinle aynı: 10 duygu)
NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative"
]

# Spotify audio feature set
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo"
]

# Model ayarları
PCA_VARIANCE = 0.90
BEST_K = 5  # ✅ K=5 deniyoruz
RANDOM_STATE = 42

# Dosyalar
ARTIFACT_PATH = "musicdna_artifacts.joblib"

# Veri temizliği
USELESS_COLS = [
    "track_album_id",
    "track_album_name",
    "playlist_id",
    "playlist_name",
    "track_album_release_date",
]


