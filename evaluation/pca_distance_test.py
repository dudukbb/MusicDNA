import joblib
import numpy as np
import matplotlib.pyplot as plt
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

art = joblib.load(ART_PATH)

df = art["df_final"]
model_features = art["model_features"]
scaler = art["scaler"]
pca = art["pca"]
kmeans = art["kmeans"]

if "cluster" not in df.columns:
    raise ValueError("df_final içinde 'cluster' kolonu yok. Önce pipeline.py çalıştırıp artifacts üretmelisin.")

labels = df["cluster"].astype(int).values

# X -> scaler -> PCA
X = df[model_features].copy()
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)  # (n_samples, n_components)

# Centroidler PCA uzayında (kmeans PCA uzayında fit edildiği için direkt)
centroids = kmeans.cluster_centers_  # (k, n_components)

# --- SAME: her noktanın kendi cluster centroid'ine uzaklığı
same_dist = np.linalg.norm(X_pca - centroids[labels], axis=1)

# --- DIFF: her noktanın en yakın diğer centroid'e uzaklığı
d_all = np.linalg.norm(X_pca[:, None, :] - centroids[None, :, :], axis=2)  # (n_samples, k)
d_all[np.arange(len(labels)), labels] = np.inf
diff_dist = np.min(d_all, axis=1)

# özet istatistikler
same_mean, same_med = float(np.mean(same_dist)), float(np.median(same_dist))
diff_mean, diff_med = float(np.mean(diff_dist)), float(np.median(diff_dist))

print(f"✅ SAME (küme içi centroid'e)  mean: {same_mean:.4f} | median: {same_med:.4f}")
print(f"✅ DIFF (en yakın diğer centroid) mean: {diff_mean:.4f} | median: {diff_med:.4f}")

ratio = same_mean / diff_mean if diff_mean > 0 else np.inf
print(f"📌 mean_ratio (same/diff): {ratio:.4f}  (küçük olması iyi)")

if ratio < 1:
    print("🟢 Güzel: küme içi mesafeler küme dışına göre daha küçük.")
else:
    print("🟠 Uyarı: küme içi mesafeler çok küçük değil; kümeler iç içe olabilir.")

# histogram
plt.figure(figsize=(10, 5))
plt.hist(same_dist, bins=40, alpha=0.7, label="Küme içi (centroid)")
plt.hist(diff_dist, bins=40, alpha=0.7, label="Küme dışı (en yakın centroid)")
plt.title("Küme İçi vs Küme Dışı PCA Mesafe Dağılımı")
plt.xlabel("PCA Mesafesi")
plt.ylabel("Frekans")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

out_path = OUT_DIR / "pca_distance_hist.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"📌 Histogram kaydedildi: {out_path}")

plt.show()

