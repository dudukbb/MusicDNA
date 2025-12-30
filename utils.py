# utils.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

def get_here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def find_csv_path() -> Path:
    """
    Repo içinde CSV bazen datasets/ altında, bazen kökte olabiliyor.
    En olası lokasyonları sırayla dener.
    """
    here = get_here()
    candidates = [
        here / "datasets" / "spotify_songs.csv",
        here / "spotify_songs.csv",
        here.parent / "datasets" / "spotify_songs.csv",
        here.parent / "spotify_songs.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "spotify_songs.csv bulunamadı. datasets/spotify_songs.csv veya proje köküne koy."
    )

def _normalize(s: str) -> str:
    return str(s).lower().strip()

def find_song_row(df: pd.DataFrame, track_name: str, track_artist: str | None = None):
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
