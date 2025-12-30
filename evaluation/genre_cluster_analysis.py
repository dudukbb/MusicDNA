import joblib
import pandas as pd
from pathlib import Path

# ===============================
# Output dizini
# ===============================
OUT_DIR = Path("evaluation/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Artifacts yolu
# ===============================
ART_PATH = "musicdna_artifacts.joblib"

# ===============================
# Artifacts yükle
# ===============================
art = joblib.load(ART_PATH)
df = art["df_final"]

if "playlist_genre" not in df.columns:
    raise ValueError("playlist_genre kolonu bulunamadı.")

print("\n===== CLUSTER x GENRE DAĞILIMI (YÜZDE) =====\n")

# ===============================
# Cluster x Genre tablosu
# ===============================
ct = pd.crosstab(
    df["cluster_name"],
    df["playlist_genre"],
    normalize="index"
).round(3)

print(ct)

# ===============================
# CSV olarak kaydet
# ===============================
out_path = OUT_DIR / "cluster_genre_table.csv"
ct.to_csv(out_path)

print(f"\n📌 Kaydedildi: {out_path}")

