# spotify.py
from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from config import ARTIFACT_PATH
from artifacts import export_artifacts, load_artifacts
from engine import music_dna_engine
from persona import seed_from_user_songs, generate_dynamic_comment
from pipeline import build_artifacts


def train_and_export(csv_path: str | None = None, out_path: str = ARTIFACT_PATH) -> None:
    """Pipeline'ı çalıştırır ve artifacts dosyasını üretir."""
    df_final, scaler, pca, kmeans, model_features, cluster_names = build_artifacts(csv_path=csv_path)

    export_artifacts(
        df_final=df_final,
        scaler=scaler,
        pca=pca,
        kmeans=kmeans,
        model_features=model_features,
        cluster_names=cluster_names,
        out_path=out_path,
    )


def run_demo(
    songs: list[dict[str, Any]],
    artifacts_path: str = ARTIFACT_PATH,
    top_n: int = 5,
) -> None:
    """Artifacts dosyasından modeli yükler ve demo çalıştırır."""
    art = load_artifacts(artifacts_path)

    dna, cluster_id, soft_top2, recs, used = music_dna_engine(
        user_song_list=songs,
        df_final=art["df_final"],
        model_features=art["model_features"],
        scaler=art["scaler"],
        pca=art["pca"],
        kmeans=art["kmeans"],
        cluster_names=art["cluster_names"],
        top_n=top_n,
    )

    print("\n===== DEMO SONUÇ =====")
    if dna is None:
        print("Şarkı bulunamadı.")
        return

    title, subtitle = (str(dna).split("\n", 1) + [""])[:2]
    print(f"MusicDNA Profilin: {title}")
    if subtitle.strip():
        print(f"Alt Başlık: {subtitle}")

    seed = seed_from_user_songs(songs)
    print("\n🔮 Kişilik Yorumu:")
    print(generate_dynamic_comment(title, seed=seed))

    if soft_top2 and len(soft_top2) >= 2:
        p1 = float(soft_top2[0]["prob"]) * 100
        p2 = float(soft_top2[1]["prob"]) * 100
        n1 = str(soft_top2[0]["cluster_name"]).split("\n", 1)[0]
        n2 = str(soft_top2[1]["cluster_name"]).split("\n", 1)[0]
        print("\n🎯 Yakınlıklar (Top-2):")
        print(f"- {n1}: %{p1:.1f}")
        print(f"- {n2}: %{p2:.1f}")

    print("\n🔗 Eşleşen şarkılar:")
    for u in used:
        print(f"- {u.get('track_name','')} — {u.get('track_artist','')}")

    print("\nDNA'na uygun tavsiyeler:")
    if recs is None or len(recs) == 0:
        print("(öneri yok)")
    else:
        cols = [c for c in ["track_name", "track_artist", "track_popularity"] if c in recs.columns]
        print(recs[cols].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="MusicDNA runner (train/export + demo)")
    p.add_argument("--train", action="store_true", help="Artifacts üret ve kaydet")
    p.add_argument("--csv", type=str, default=None, help="CSV yolu (opsiyonel)")
    p.add_argument("--out", type=str, default=ARTIFACT_PATH, help="Artifact output path")

    p.add_argument("--demo", action="store_true", help="Demo çalıştır (artifact dosyasından)")
    p.add_argument("--artifact", type=str, default=ARTIFACT_PATH, help="Kullanılacak artifact path")
    p.add_argument("--top-n", type=int, default=5, help="Demo öneri sayısı")

    p.add_argument(
        "--song",
        action="append",
        default=[],
        help='Demo şarkısı. Format: "Anaconda|Nicki Minaj" veya "Shape of You|"',
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.train:
        train_and_export(csv_path=args.csv, out_path=args.out)

    if args.demo:
        if args.song:
            songs: list[dict[str, Any]] = []
            for raw in args.song:
                name, artist = (raw.split("|", 1) + [""])[:2] if "|" in raw else (raw, "")
                songs.append({"name": name.strip(), "artist": artist.strip() or None})
        else:
            songs = [
                {"name": "Anaconda", "artist": "Nicki Minaj"},
                {"name": "Shape of You", "artist": "Ed Sheeran"},
            ]
        run_demo(songs=songs, artifacts_path=args.artifact, top_n=args.top_n)


if __name__ == "__main__":
    main()
