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
    top_n=5,
):
    found_rows = []
    for s in user_song_list:
        row = find_song_row(df_final, s.get("name"), s.get("artist"))
        if row is not None:
            found_rows.append(row)

    # ✅ 5'li dönüş (dna, cluster_id, soft_top2, recs, used)
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
    centers = kmeans.cluster_centers_  # (k, pca_dim)
    dists = np.linalg.norm(centers - user_pca[0], axis=1)  # (k,)

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

    used_names = {r["track_name"].lower() for r in found_rows if isinstance(r.get("track_name"), str)}
    recs = df_final[df_final["cluster"] == cluster_id].copy()
    recs = recs[~recs["track_name"].str.lower().isin(used_names)]

    top_recs = recs.sort_values("track_popularity", ascending=False).head(top_n)

    used_pretty = [
        {"track_name": r["track_name"], "track_artist": r["track_artist"]}
        for r in found_rows
    ]

    return (
        dna_name,
        cluster_id,
        soft_top2,
        top_recs[["track_name", "track_artist", "track_popularity"]],
        used_pretty,
    )
