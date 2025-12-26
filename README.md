# MusicDNA 🎧

Inferring personality-related listening tendencies using Spotify audio features and lyrics-based emotion signals (NRCLex) with unsupervised learning (PCA + KMeans).

## Project Overview
MusicDNA takes a small list of songs from a user and:
- extracts audio + lyrics-based emotion features
- reduces dimensionality with PCA
- clusters songs with KMeans
- generates an interpretable “music personality” summary for the user based on the closest cluster(s)

## Dataset
This project uses the publicly available Kaggle dataset:

- **Audio Features and Lyrics of Spotify Songs**
  - https://www.kaggle.com/datasets/muhammad/audio-features-and-lyrics-of-spotify-songs

The dataset includes Spotify audio features, lyrics, genre/subgenre metadata, and other song information.

## Features Used
We build clustering features from two layers:
- **Audio features** (Spotify numeric features)
- **Lyrics emotions** from **NRCLex** (e.g., joy, sadness, anger, fear, trust...)

> Note: `genre` / `subgenre` are used **only for interpretation and labeling**, not as clustering inputs.

## Method
1. **Preprocessing**
   - handle missing values
   - scale numeric features (StandardScaler)
2. **Lyrics Emotion Extraction**
   - NRCLex is applied to lyrics and produces emotion scores per song
3. **Dimensionality Reduction**
   - PCA with explained variance threshold
4. **Clustering**
   - KMeans (best_k determined via elbow / silhouette experiments)
5. **Cluster Naming**
   - cluster profiles are summarized using z-score comparisons and assigned human-friendly names

## Project Structure


