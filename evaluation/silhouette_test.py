import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, silhouette_samples

# ===============================
# Output dizini
# ===============================
OUT_DIR = Path("evaluation/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Artifacts yolu
# ===============================
ART_PATH = "musicdna_artifacts.joblib"

art = joblib.load(ART_PATH)

df = art["df_final"]
model_features = art["model_features"]
scaler = art["scaler"]
pca = art["pca"]

# labels: df_final içindeki cluster sütunu
if "cluster" not in df.columns:
    raise ValueError("df_final içinde 'cluster' kolonu yok. Önce pipeline.py ile artifacts üretmelisin.")

labels = df["cluster"].astype(int).values

# X -> scaler -> PCA
X = df[model_features].copy()
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

print("X_pca shape:", X_pca.shape)  # kontrol

sil = silhouette_score(X_pca, labels)
print(f"✅ Silhouette Score: {sil:.4f}")

# örnek bazlı silhouette dağılımı
samps = silhouette_samples(X_pca, labels)

plt.figure(figsize=(10, 5))
plt.hist(samps, bins=40)
plt.title("Silhouette Dağılımı")
plt.xlabel("Silhouette değeri")
plt.ylabel("Frekans")
plt.grid(alpha=0.2)
plt.tight_layout()

out_path = OUT_DIR / "silhouette_hist.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"📌 Histogram kaydedildi: {out_path}")

plt.show()

