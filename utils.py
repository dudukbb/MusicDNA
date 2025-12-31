# utils.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATASET_FILENAME


def get_here() -> Path:
    """Bu dosyanın bulunduğu dizini döndürür (notebook ortamında cwd)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def find_csv_path(filename: str = DATASET_FILENAME) -> Path:
    """
    Repo içinde CSV bazen datasets/ altında, bazen kökte olabiliyor.
    En olası lokasyonları sırayla dener ve ilk bulduğunu döndürür.
    """
    here = get_here()
    candidates = [
        here / "datasets" / filename,
        here / filename,
        here.parent / "datasets" / filename,
        here.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"CSV bulunamadı: {filename}. Denenen yollar: {[str(c) for c in candidates]}"
    )


def _normalize(text: str) -> str:
    """Basit normalize: lower, strip, unicode normalize, çoklu boşluk tek boşluk."""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _series_contains(series: pd.Series, needle: str) -> pd.Series:
    """NaN güvenli, regex kapalı contains."""
    return series.astype(str).str.lower().str.contains(needle, na=False, regex=False)


def find_song_row(
    df: pd.DataFrame,
    track_name: str,
    track_artist: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    track_name ile contains (regex=False) arar.
    track_artist verilirse sonuçları sanatçıya göre filtreler.
    track_popularity varsa en popüler olanı döndürür.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if not track_name or not isinstance(track_name, str) or track_name.strip() == "":
        return None
    if "track_name" not in df.columns:
        return None

    tn = _normalize(track_name)
    hits = df[_series_contains(df["track_name"], tn)].copy()
    if hits.empty:
        return None

    if track_artist and isinstance(track_artist, str) and track_artist.strip() != "":
        if "track_artist" in hits.columns:
            ta = _normalize(track_artist)
            artist_mask = _series_contains(hits["track_artist"], ta)
            if artist_mask.any():
                hits = hits[artist_mask]

    if "track_popularity" in hits.columns:
        hits = hits.sort_values("track_popularity", ascending=False)

    return hits.iloc[0]

