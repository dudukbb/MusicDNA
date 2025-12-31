# engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import find_song_row


def music_dna_engine(
    user_song_list,
    df_final,
    model_features,
    scaler,
    pca,
    kmeans,
    cluster_names,
    top_n: int = 5,
):
    """
    ✅ 5'li dönüş:
      (dna_name, cluster_id, soft_top2, recs_df, used_pretty)
    """
    found_rows = []
    for s in user_song_list:
        row = find_song_row(df_final, s.get("name"), s.get("artist"))
        if row is not None:
            found_rows.append(row)

    if not found_rows:
        return None, None, None, None, []

    user_vec = pd.DataFrame(found_rows)[model_features].mean().to_frame().T
    user_scaled = scaler.transform(user_vec)
    user_pca = pca.transform(user_scaled)

    cluster_id = int(kmeans.predict(user_pca)[0])
    dna_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    # -----------------------------
    # Soft assignment (top-2 yakınlık)
    # -----------------------------
    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(centers - user_pca[0], axis=1)

    temperature = 1.0
    scores = np.exp(-dists / temperature)
    probs = scores / scores.sum()

    top2 = probs.argsort()[-2:][::-1]
    soft_top2 = [
        {
            "cluster_id": int(c),
            "cluster_name": cluster_names.get(int(c), f"Cluster {int(c)}"),
            "prob": float(probs[c]),
            "dist": float(dists[c]),
        }
        for c in top2
    ]

    used_names = {
        r["track_name"].lower()
        for r in found_rows
        if isinstance(r.get("track_name"), str)
    }

    recs = df_final[df_final["cluster"] == cluster_id].copy()
    if "track_name" in recs.columns:
        recs = recs[~recs["track_name"].astype(str).str.lower().isin(used_names)]

    sort_col = "track_popularity" if "track_popularity" in recs.columns else None
    if sort_col:
        recs = recs.sort_values(sort_col, ascending=False)

    cols = [c for c in ["track_name", "track_artist", "track_popularity"] if c in recs.columns]
    top_recs = recs[cols].head(top_n) if cols else recs.head(top_n)

    used_pretty = [
        {"track_name": r.get("track_name"), "track_artist": r.get("track_artist")}
        for r in found_rows
    ]

    return dna_name, cluster_id, soft_top2, top_recs, used_pretty
