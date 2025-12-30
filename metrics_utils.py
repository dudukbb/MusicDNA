# metrics_utils.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def plot_corr_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    title: str,
    figsize=(10, 8),
    annot: bool = False,
):
    """Basit korelasyon heatmap (matplotlib)."""
    if not cols:
        print("[corr] Kolon listesi boş.")
        return

    data = df[cols].copy()
    corr = data.corr(numeric_only=True)

    plt.figure(figsize=figsize)
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return corr


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score (en az 2 cluster olmalı)."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def intra_cluster_distance(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Basit intra-cluster mesafe:
    Her noktanın kendi cluster centroid'ine olan ÖKLİD ortalaması.
    """
    uniq = np.unique(labels)
    cents = []
    for c in uniq:
        pts = X[labels == c]
        cents.append(pts.mean(axis=0))
    cents = np.vstack(cents)

    # label -> centroid
    cent_map = {c: cents[i] for i, c in enumerate(uniq)}
    dists = []
    for i in range(X.shape[0]):
        dists.append(np.linalg.norm(X[i] - cent_map[labels[i]]))
    return float(np.mean(dists))


def inter_centroid_distance(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Basit inter-cluster mesafe:
    Centroidler arası tüm çiftlerin ÖKLİD ortalaması.
    """
    uniq = np.unique(labels)
    cents = []
    for c in uniq:
        pts = X[labels == c]
        cents.append(pts.mean(axis=0))
    cents = np.vstack(cents)

    if cents.shape[0] < 2:
        return float("nan")

    pair = []
    for i in range(cents.shape[0]):
        for j in range(i + 1, cents.shape[0]):
            pair.append(np.linalg.norm(cents[i] - cents[j]))
    return float(np.mean(pair))
