# export_artifacts.py
import joblib

# Bu dosyayı çalıştırmadan önce:
# df_final, scaler, pca, kmeans, model_features, cluster_names objelerinin
# bellekte hazır olması gerekir.
#
# En pratik kullanım:
# - eğitim kodunu çalıştır (spotify.py / notebook)
# - en sonda bu export kodunu çağır

def export_artifacts(
    df_final,
    scaler,
    pca,
    kmeans,
    model_features,
    cluster_names,
    out_path="musicdna_artifacts.joblib",
):
    payload = {
        "df_final": df_final,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "model_features": model_features,
        "cluster_names": cluster_names,
    }
    joblib.dump(payload, out_path)
    print(f"✅ Kaydedildi: {out_path}")


# Eğer bunu direkt çalıştırmak istersen, burada df_final vb. objeleri import etmen gerekir.
# Ama genelde sen eğitim scriptinin sonunda export_artifacts(...) diye çağırırsın.
if __name__ == "__main__":
    print("Bu dosya tek başına çalıştırılacaksa önce df_final/scaler/pca/kmeans objeleri oluşturulmalı.")
