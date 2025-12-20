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
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.cluster import kmeans
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
        sns.countplot(x=dataframe[col_name], data=dataframe)
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
    # tipi Object olanlar
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # numeric görünümlü categoric ( sayısal gibi ama aslında categoric --> mode = {0,1}
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    # categoric ama cardinal
    # categoric gibi ama çok fazla sınıfı var(cardinal)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    # tipi obejct + numeric görünümlü categoricler = gerçek categorics
    cat_cols = cat_cols + num_but_cat
    # categoriclerin içinden cat_but_carları çıkarıyoruz
    # çünkü cat_but_car categoric görünümlü cardinal olduğu için çıkarmamız gerekiyor
    # cat_but_car : gerçekten categoric değil, o cardinal
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # object olmayan tüm featurelar(kolonlar)
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # numeric görünümlü categoricleri çıkarıyoruz = gerçek numericler
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

df = pd.read_csv("spotify_songs.csv")
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
num_cols.remove("key")
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
# bu sayıları toplan kelime sayısına bölüyoruz ( denom )
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
# kullanıcının aradığı kelimeyi içeren tüm satırları bulur, ilgilim tüm şarkıları hits adındaki listeye alır
# kullanıcı sanatçı adı da vermişse bulduğu sonuçları bu sanatçıya göre tekrar filtreler. Böylece doğru şarkıyı vulma ihtimali artar.
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
# bu sözlükleri tolist() ile listeye çevirip sonra bir df halien getiriyoruz.
lyrics_feat_df = pd.DataFrame(lyrics_results.tolist())
actual_lyrics_features = [c for c in lyrics_feat_df.columns if c.startswith("lyr_")]

# df & lyrics_feat_df merge işlemi
df_final = pd.concat([df.reset_index(drop=True),
                      lyrics_feat_df.reset_index(drop=True)], axis=1)


# modele girecek olan tüm numeric sütunları tek bir listede topluyoruz
model_features = audio_features + actual_lyrics_features
check_df(df,head=5)

# modelde kullanılacak numeric kolonları seçtik df içinden
X = df_final[model_features]

# Standartlaştırma objesi
scaler = StandardScaler()

# verilere standartlaştırıyoruz ( avg=0, std=1 )
# veriler aynı ölçekte olmalı, biri diğerini bastırmamalı
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. MODELLEME (K-MEANS & PCA)
# ==========================================
# optimal k değeri için Elbow Methodunu kullandık
ssd = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    ssd.append(km.inertia_)

# Elbow Grafiği Görselleştirme
plt.figure(figsize=(10, 5))
plt.plot(K_range, ssd, "bx-", markersize=8, linewidth=2)
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Inertia (SSD)")
plt.title("İdeal Küme Sayısı İçin Dirsek Metodu (Elbow)")
plt.grid(True)
plt.show()

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
def name_cluster(z):
    tags = []

    # 1. PSİKOLOJİK KATMAN: Russell'ın Circumplex Modeli (Energy & Valence Dengesi)
    # Energy (Arousal) ve Valence (Pleasure) eksenlerine göre duygu durumu tayini
    if z["energy"] > 0 and z["valence"] > 0:
        tags.append("Mutlu & Coşkulu")
    elif z["energy"] > 0 and z["valence"] <= 0:
        tags.append("Agresif & Dinamik")
    elif z["energy"] <= 0 and z["valence"] <= 0:
        tags.append("Hüzünlü & Melankolik")
    elif z["energy"] <= 0 and z["valence"] > 0:
        tags.append("Huzurlu & Sakin")

    # 2. TEKNİK KATMAN: Ses Karakteristiği (Audio Features)
    if z["danceability"] > 0.5:
        tags.append("Dans/Party")
    if z["instrumentalness"] > 0.6:
        tags.append("Enstrümantal/Odak")
    if z["speechiness"] > 0.6:
        tags.append("Sözel/Rap")
    if z["acousticness"] > 0.5:
        tags.append("Akustik")

    # 3. LYRICAL KATMAN: Sözlerdeki Duygu Derinliği (Lyrics Features)
    # Eğer psikolojik katman "Hüzünlü" dediyse ve sözler de bunu destekliyorsa pekiştirir
    if z["lyr_sadness"] > 0.4 or z["lyr_negative"] > 0.4:
        if "Hüzünlü & Melankolik" not in tags:  # Tekrarı önlemek için
            tags.append("Duygusal Derinlik")
    if z["lyr_joy"] > 0.4 or z["lyr_positive"] > 0.4:
        if "Mutlu & Coşkulu" not in tags:
            tags.append("Pozitif Vibes")

    # En belirgin ilk 2 etiketi birleştir, yoksa "Dengeli Karma" de
    return " + ".join(tags[:2]) if tags else "Dengeli / Karma"


# Uygulama ve Eşleme
cluster_names = cluster_z.apply(name_cluster, axis=1).to_dict()
df_final["cluster_name"] = df_final["cluster"].map(cluster_names)

# Sonuçları Kontrol Et
print("\n===== MÜZİKAL KİŞİLİK DAĞILIMI =====")
print(df_final["cluster_name"].value_counts())

# ==========================================
# 6. GÖRSELLEŞTİRME (PCA 2D)
# ==========================================
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)
df_final["pca1"], df_final["pca2"] = X_pca2[:, 0], X_pca2[:, 1]
centroids_2d = pca2.transform(kmeans.cluster_centers_)

plt.figure(figsize=(12, 7))
scatter = plt.scatter(df_final["pca1"], df_final["pca2"], c=df_final["cluster"], cmap='viridis', alpha=0.3)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=300, color='red')

for i, (x, y) in enumerate(centroids_2d):
    plt.text(x, y, f"Cluster {i}: {cluster_names[i]}", fontsize=9, fontweight="bold", backgroundcolor='white')

plt.title("MusicDNA Kümeleme Haritası")
plt.show()


# ==========================================
# 7. KULLANICI PROFİLLEME VE TAVSİYE
# ==========================================
def music_dna_engine(user_song_list):
    found_rows = []
    for s in user_song_list:
        row = find_song_row(df_final, s.get('name'), s.get('artist'))
        if row is not None: found_rows.append(row)

    if not found_rows: return "Şarkı bulunamadı.", None

    user_vec = pd.DataFrame(found_rows)[model_features].mean().to_frame().T
    user_scaled = scaler.transform(user_vec)
    cluster_id = int(kmeans.predict(user_scaled)[0])
    dna_name = cluster_names[cluster_id]

    recs = df_final[df_final["cluster"] == cluster_id].copy()
    recs = recs[~recs["track_name"].str.lower().isin([r['track_name'].lower() for r in found_rows])]
    top_recs = recs.sort_values("track_popularity", ascending=False).head(5)

    return dna_name, top_recs[["track_name", "track_artist"]]


# --- ÖRNEK KULLANIM ---
demo_songs = [{"name": "Anaconda", "artist": "Nicki Minaj"}, {"name": "Shape of You"}]
dna, recommendations = music_dna_engine(demo_songs)

print(f"\n===== SONUÇ =====")
print(f"Sizin Müzikal Kişiliğiniz: {dna}")
print("\nDNA'nıza Uygun Tavsiyeler:")
print(recommendations)

