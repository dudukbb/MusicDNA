# ===================== VERİ SETİ ÖZELLİK AÇIKLAMALARI =====================
# track_id                     : Spotify şarkı için benzersiz kimlik (ID)
# track_name                   : Şarkının adı
# track_artist                 : Şarkının ana sanatçısı
# lyrics                       : Şarkı sözleri (yalnızca duygu/emosyon çıkarımı için kullanılır, sonra çıkarılır)
#
# track_popularity              : Spotify tarafından hesaplanan popülerlik skoru (0–100, zamana bağlı)
# track_album_id               : Albüm için benzersiz Spotify kimliği (ID)
# track_album_name             : Albüm adı
# track_album_release_date     : Albüm çıkış tarihi (string formatında, modellemede kullanılmaz)
#
# playlist_name                : Şarkının bulunduğu Spotify çalma listesinin adı
# playlist_id                  : Spotify çalma listesi benzersiz kimliği (ID)
# playlist_genre               : Çalma listesinin ana türü (ör. Pop, Rock)
# playlist_subgenre            : Çalma listesinin alt türü (daha detaylı tür etiketi)
#
# ===================== SPOTIFY SES (AUDIO) ÖZELLİKLERİ =====================
# danceability                : Şarkının dans edilebilirlik seviyesi (0.0–1.0)
# energy                      : Şarkının algılanan enerji ve hareketlilik seviyesi (0.0–1.0)
# key                         : Şarkının müzikal tonu (0=Do, 1=Do#, ..., 11=Si) – kategorik olarak ele alınır
# loudness                    : Şarkının genel ses yüksekliği (desibel cinsinden, genellikle -60 ile 0 arası)
# mode                        : Şarkının modu (1=Majör, 0=Minör)
# speechiness                 : Şarkıda konuşma ağırlıklı bölümlerin oranı (0.0–1.0)
# acousticness                : Şarkının akustik olma olasılığı (0.0–1.0)
# instrumentalness            : Şarkının vokal içermeme olasılığı (0.0–1.0)
# liveness                    : Şarkının canlı performans olma olasılığı (0.0–1.0)
# valence                     : Şarkının aktardığı müzikal pozitiflik seviyesi (0.0–1.0)
# tempo                       : Şarkının tahmini temposu (BPM – dakikadaki vuruş sayısı)
# duration_ms                 : Şarkı süresi (milisaniye cinsinden)
#
# ===================== DİL VE METAVERİ =====================
# language                    : Şarkı sözlerinin tespit edilen dili
#
# ========================================================================

######################################## MusicDNA Pipeline #######################################
# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function
###################################################################################################
# bazi değişkenlerdeki NA'ler 0 oluyor buna dikkat

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#from nltk.cluster import kmeans
from nrclex import NRCLex
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#############################################
# HELPER FUNCTION
#############################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Duplicates #####################")
    print("Total Duplicates:", dataframe.duplicated().sum())
    if "track_id" in dataframe.columns:
        print("Track ID Duplicates:", dataframe["track_id"].duplicated().sum())

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=3, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # cat_cols ile num_cols çakışmasın (numeric olanlar cat_cols'ta kalmasın)
    cat_cols = [col for col in cat_cols if col not in num_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/spotify_songs.csv")
df.head()
df.dtypes
check_df(df,head=5)

# lyrics ve language'daki NA değerleri drop eder
df.dropna(subset=["lyrics", "language"], inplace=True)

# NA var mı ?
df.isnull().sum()
check_df(df,head=5)

# işimize yaramayacak columları sileriz
useless_cols = [
    "track_album_id",
    "track_album_name",
    "playlist_id",
    "playlist_name",
    "track_album_release_date"
]
df.drop(useless_cols, axis=1, inplace=True)
check_df(df,head=5)

df.dtypes
# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=3, car_th=20)
num_cols = [col for col in num_cols if col not in ["track_popularity", "duration_ms"]]

# key threshold=5 olayından dolayı
# key -----> num görünümlü categoric
if "key" in num_cols:
    num_cols.remove("key")
if "key" not in num_but_cat:
    num_but_cat.append("key")

print(num_cols)
print(cat_cols)
print(num_but_cat)
print(cat_but_car)

df.head()

# categoric değişken inceleme
for col in cat_cols:
    cat_summary(df,col)

# numeric değişken inceleme
df[num_cols].describe().T

# numeric değişkenlerin grafiğini oluşturmak istersek
for col in num_cols:
    num_summary(df, col, plot=True)

# numeric değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)

# OUTLİER kontrolü yaptık ama outlierları yok etmedik
# çünkü kişilik analizi yapacağız ve her uç veri bizim için önemli

#####################################################
# NRCLex & Helper Functionlar
#####################################################

# tarckin ses özellikleri
audio_features = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo"
]

# lyrics'ten çıkaracağımız 10 temel duygu
NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative"
]

# bir metin alınıp içindeki duygu yüklü kelimeleri sayıyoruz
# bu sayıları toplam kelime sayısına bölüyoruz ( denom )
# NEDEN ? -uzun şarkılar(örneğin Rap) kısa şarkılardan daha fazla
# duygu kelimesi içerebilir. Kelime sayısına bölerek(normalization)
# her şarkının "duygu yoğunluğunu" ölçüyoruz.
def nrclex_features(text: str) -> dict:
    """Lyrics metninden duygu skorları çıkarır."""
    if not isinstance(text, str) or text.strip() == "":
        return {f"lyr_{e}": 0.0 for e in NRC_EMOTIONS}
    lex = NRCLex(text)
    raw = lex.raw_emotion_scores
    token_count = len(lex.words) if hasattr(lex, "words") else 0
    denom = token_count if token_count > 0 else 1
    return {f"lyr_{e}": raw.get(e, 0) / denom for e in NRC_EMOTIONS}

# input olarak aldığı metni ( şarkı veya sanatçı adı) küçük harfe çevirip
# başındaki ve sonundaki gereksiz boşlukları siliyor.
# arama sırasında hatayı minimize etmek için yapıyoruz.
def _normalize(s: str) -> str:
    return str(s).lower().strip()

# kullanıcının yazdığı şarkı ismini _normalize fonksiyonuna göndererek temizler.
# kullanıcının aradığı kelimeyi içeren tüm satırları bulur, ilgili tüm şarkıları hits adındaki listeye alır
# kullanıcı sanatçı adı da vermişse bulduğu sonuçları bu sanatçıya göre tekrar filtreler. Böylece doğru şarkıyı bulma ihtimali artar.
# eğer aynı isimde birden fazla kayıt varsa(mesela bir şarkının hem orijinali hem de remixi) en popülerden en az popülere göre sıralar.
# sıralamanın en başındaki (.iloc[0]), yani en popüler olan satırı seçer ve bu şarkının tüm verilerini return eder.
def find_song_row(df, track_name, track_artist=None):
    if not track_name: return None
    tn = _normalize(track_name)
    mask = df["track_name"].str.lower().str.contains(tn, na=False, regex=False)
    hits = df[mask].copy()
    if hits.empty: return None
    if track_artist:
        ta = _normalize(track_artist)
        artist_mask = hits["track_artist"].str.lower().str.contains(ta, na=False, regex=False)
        if artist_mask.any(): hits = hits[artist_mask]
    return hits.sort_values("track_popularity", ascending=False).iloc[0]


# ==========================================
#  BİRLEŞTİRME ÖNCESİ PREPROCESSING
# ==========================================
# Duplicate Temizliği
df = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)

# Duygu Analizi Hesaplama
print("Adım 1: Lyrics Duygu Analizi yapılıyor ...")
# apply ile her bir şarkı sözünü nrclex_features fonksiyonua gönderiyoruz.
lyrics_results = df["lyrics"].apply(nrclex_features)

# nrclex_features fonksiyonu her şarkı için bir dictionary döndürür.
# bu sözlükleri tolist() ile listeye çevirip sonra bir df haline getiriyoruz.
lyrics_feat_df = pd.DataFrame(lyrics_results.tolist())
actual_lyrics_features = [c for c in lyrics_feat_df.columns if c.startswith("lyr_")]

# df & lyrics_feat_df merge işlemi
df_final = pd.concat([df.reset_index(drop=True),
                      lyrics_feat_df.reset_index(drop=True)], axis=1)


# modele girecek olan tüm numeric sütunları tek bir listede topluyoruz
model_features = audio_features + actual_lyrics_features
check_df(df_final,head=5)


# modelde kullanılacak numeric kolonları seçtik df içinden
X = df_final[model_features].copy()

# Standartlaştırma objesi
scaler = StandardScaler()

# verilere standartlaştırıyoruz ( avg=0, std=1 )
# veriler aynı ölçekte olmalı, biri diğerini bastırmamalı
X_scaled = scaler.fit_transform(X)


#PCA (varyansın %90'ını koru)
# n_components=0.90 -> "toplam varyansın %90'ını açıklayan bileşen sayısını otomatik seç"
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Orijinal feature sayısı:", X.shape[1])
print("PCA sonrası bileşen sayısı:", X_pca.shape[1])
print("Açıklanan toplam varyans oranı:", pca.explained_variance_ratio_.sum())

# ==========================================
# 4. MODELLEME (K-MEANS & PCA)
# ==========================================
# optimal k değeri için Elbow Methodunu kullandık
ssd = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    ssd.append(km.inertia_)

# Elbow Grafiği Görselleştirme
plt.figure(figsize=(10, 5))
plt.plot(K_range, ssd, "bx-", markersize=8, linewidth=2)
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Inertia (SSD)")
plt.title("İdeal Küme Sayısı İçin Dirsek Metodu (Elbow)")
plt.grid(True)
plt.show()

best_k = 4  # senin seçtiğin k (elbow/silhouette'tan gelen)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

df_final["cluster"] = kmeans.fit_predict(X_pca)

df_final["cluster"].value_counts()
# ==========================================
# 5. KİŞİLİK ANALİZİ VE İSİMLENDİRME
# ==========================================
#Ne yapılıyor?: Her kümenin (0, 1, 2, 3, 4) müzikal özelliklerinin ortalamasını alıyoruz.
# Ardından bu ortalamaları kendi içinde tekrar standartlaştırıyoruz (Z-skoru).
# NEDEN ? : Bir kümenin "Mutlu" olduğunu söylemek için, onun mutluluk skorunun
# sadece yüksek olması yetmez; diğer kümelere kıyasla belirgin şekilde yüksek olması gerekir.
# Z-skoru bize "kim ortalamadan ne kadar saptı" bilgisini verir.
cluster_profile = df_final.groupby("cluster")[model_features].mean()
cluster_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
cluster_z = cluster_z.replace([np.inf, -np.inf], np.nan).fillna(0)

# İsimlendirme Mantığı
# Psikolojik Katman (Russell'ın Modeli): Müziğin enerjisi (Energy) ve neşesi (Valence) üzerinden temel ruh halini belirler.
# Yüksek Enerji + Yüksek Valans = Mutlu & Coşkulu
# Düşük Enerji + Düşük Valans = Hüzünlü & Melankolik
# Teknik Katman: Şarkının fiziksel yapısını kontrol eder. Çok fazla kelime varsa "Sözel/Rap", enstrüman ağırlıklıysa "Enstrümantal/Odak" etiketini yapıştırır.
# Lyrical (Sözsel) Katman: NLP analizi sonuçlarını kontrol eder. Eğer şarkı sözlerinde hüzün kelimeleri (sadness) diğer kümelere göre çok baskınsa "Duygusal Derinlik" ekler.
def name_cluster(
    z,
    max_tags=3,
    top_k=2,
    min_z=0.8,
):
    """
    z: cluster'ın z-skor profil sözlüğü (örn. cluster_z.loc[cid].to_dict()).
       Z-skor: 0 = ortalama, 1 = ~1 std üstü.

    Strateji:
    1) Psikolojik katman (Energy & Valence) -> her zaman 1 ana mood etiketi
    2) Teknik katman -> seçilebilir adaylar (danceability, instrumentalness, speechiness, acousticness)
    3) Top-k: z-skoru en yüksek olanlardan min_z üstündekileri seç
    4) Toplam etiket sayısı max_tags'ı geçmesin
    """

    tags = []

    # =========================================================
    # 1) PSİKOLOJİK KATMAN (her zaman 1 etiket)
    # =========================================================
    energy = float(z.get("energy", 0.0))
    valence = float(z.get("valence", 0.0))

    if energy > 0 and valence > 0:
        tags.append("Mutlu & Coşkulu")
    elif energy > 0 and valence <= 0:
        tags.append("Agresif & Dinamik")
    elif energy <= 0 and valence <= 0:
        tags.append("Hüzünlü & Melankolik")
    else:
        tags.append("Huzurlu & Sakin")

    # Etiket kotası dolduysa erken çık
    if len(tags) >= max_tags:
        return tags[:max_tags]

    # =========================================================
    # 2) TEKNİK KATMAN (aday etiketler)
    #    Not: burada z-skorlar "hangi özellik cluster'da baskın?" sorusuna yanıt verir.
    # =========================================================
    candidates = [
        ("danceability", "Dans/Party"),
        ("instrumentalness", "Enstrümantal/Odak"),
        ("speechiness", "Sözel/Rap"),
        ("acousticness", "Akustik"),
    ]

    scored = []
    for key, label in candidates:
        val = float(z.get(key, 0.0))
        scored.append((val, label))

    # En yüksek z-skorlu adayları başa getir
    scored.sort(key=lambda x: x[0], reverse=True)

    # =========================================================
    # 3) Top-k seçimi + min_z filtresi + max_tags limiti
    # =========================================================
    remaining_slots = max_tags - len(tags)
    take = min(top_k, remaining_slots)

    picked = 0
    for score, label in scored:
        if picked >= take:
            break
        if score >= min_z:
            tags.append(label)
            picked += 1

    # Eğer min_z yüzünden hiç seçemediyse ama yer varsa:
    # (cluster çok "ortalama" ise yine de açıklayıcı 1 teknik etiket ver)
    if picked == 0 and remaining_slots > 0:
        tags.append(scored[0][1])  # en yüksek olanı ekle

    return tags[:max_tags]

# Uygulama ve Eşleme
# cluster_z: index = cluster_id, kolonlar = feature z-skorları
# name_cluster: yeni fonksiyonun (max_tags/top_k/min_z parametreli)

cluster_names = (
    cluster_z.apply(
        lambda row: " / ".join(
            name_cluster(
                row.to_dict(),
                max_tags=3,   # toplam etiket limiti
                top_k=2,      # teknik katmandan seçilecek sayi
                min_z=0.8     # "belirgin" sayılacak z eşiği
            )
        ),
        axis=1
    )
    .to_dict()
)

df_final["cluster_name"] = df_final["cluster"].map(cluster_names)

# Sonuçları Kontrol Et
print("\n===== MÜZİKAL KİŞİLİK DAĞILIMI =====")
print(df_final["cluster_name"].value_counts())

# ==========================================
# 6. GÖRSELLEŞTİRME (PCA 2D)
# ==========================================
# Modelde kullandığın PCA (0.90) zaten var: pca, X_pca
# 2D için aynı uzaydan tekrar PCA (veya direkt ilk 2 bileşeni kullan)

pca_viz = PCA(n_components=2, random_state=42)
X_viz = pca_viz.fit_transform(X_pca)

df_final["pca1"], df_final["pca2"] = X_viz[:, 0], X_viz[:, 1]
centroids_2d = pca_viz.transform(kmeans.cluster_centers_)

COLOR_MAP = {
    0: "#1f77b4",  # mavi
    1: "#ff7f0e",  # turuncu
    2: "#2ca02c",  # yeşil
    3: "#d62728",  # kırmızı
    4: "#9467bd",  # mor (5 cluster varsa)
}

plt.figure(figsize=(13, 8))
clusters = sorted(df_final["cluster"].unique())

for cid in clusters:
    sub = df_final[df_final["cluster"] == cid]
    plt.scatter(
        sub["pca1"], sub["pca2"],
        s=14,                    # biraz daha büyük nokta
        alpha=0.35,              # daha net görünür
        color=COLOR_MAP.get(cid, "#333333"),
        edgecolors="none",
        label=f"{cid} - {cluster_names[cid]}"
    )

# Centroid'ler
plt.scatter(
    centroids_2d[:, 0], centroids_2d[:, 1],
    marker="X",
    s=320,
    color="black",
    edgecolor="white",
    linewidth=1.5,
    label="Centroid"
)

# Centroid numaraları
for i, (x, y) in enumerate(centroids_2d):
    plt.text(
        x + 0.15, y + 0.15,
        f"{i}",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.95,
            edgecolor="black",
            boxstyle="round,pad=0.25"
        )
    )

plt.title("MusicDNA Kümeleme Haritası (PCA 2D)", fontsize=14, fontweight="bold")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")

# Outlier kırpma (sunum için çok iyi)
plt.xlim(df_final["pca1"].quantile(0.01), df_final["pca1"].quantile(0.99))
plt.ylim(df_final["pca2"].quantile(0.01), df_final["pca2"].quantile(0.99))

# Legend'i sade ve net yap
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    fontsize=10
)

plt.grid(alpha=0.15)   # çok hafif grid (okunabilirlik)
plt.tight_layout()
plt.show()

# ==========================================
# 7. KULLANICI PROFİLLEME VE TAVSİYE
# ==========================================
def music_dna_engine(user_song_list, df=df_final, top_n=5):
    found_rows = []
    for s in user_song_list:
        row = find_song_row(df, s.get("name"), s.get("artist"))
        if row is not None:
            found_rows.append(row)  # row: pandas Series

    if not found_rows:
        return None, None

    # Kullanıcı vektörü: bulunan şarkıların feature ortalaması
    user_vec = pd.DataFrame(found_rows)[model_features].mean().to_frame().T

    # Aynı preprocessing pipeline: scaler -> PCA -> KMeans
    user_scaled = scaler.transform(user_vec)
    user_pca = pca.transform(user_scaled)          # <-- KRİTİK (model PCA ile eğitildi)
    cluster_id = int(kmeans.predict(user_pca)[0])

    dna_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    # Öneriler: aynı cluster'dan popüler şarkılar
    used_names = {r["track_name"].lower() for r in found_rows if isinstance(r.get("track_name"), str)}
    recs = df[df["cluster"] == cluster_id].copy()

    recs = recs[~recs["track_name"].str.lower().isin(used_names)]
    top_recs = recs.sort_values("track_popularity", ascending=False).head(top_n)

    return dna_name, top_recs[["track_name", "track_artist", "track_popularity"]]

# --- ÖRNEK KULLANIM ---
demo_songs = [{"name": "Anaconda", "artist": "Nicki Minaj"}, {"name": "Shape of You"}]
dna, recommendations = music_dna_engine(demo_songs)

print("\n===== SONUÇ =====")

if dna is None:
    print("Şarkı bulunamadı. Lütfen şarkı adlarını kontrol edin.")
else:
    print(f"Sizin Müzikal Kişiliğiniz: {dna}")
    print("\nDNA'nıza Uygun Tavsiyeler:")
    print(recommendations)

from export_artifacts import export_artifacts

export_artifacts(
    df_final=df_final,
    scaler=scaler,
    pca=pca,
    kmeans=kmeans,
    model_features=model_features,
    cluster_names=cluster_names,
    out_path="musicdna_artifacts.joblib",
)
