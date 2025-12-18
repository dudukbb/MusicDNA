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
# ===================== DİL =====================
# language                    : Şarkı sözlerinin tespit edilen dili
#
# ========================================================================

##########################################################################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

# bazi değişkenlerdeki NA'ler 0 oluyor buna dikkat

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
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
df.drop("track_album_release_date", axis=1, inplace=True)

# NA var mı ?
df.isnull().sum()
check_df(df,head=5)

#####################################################
# drop_cols = [
#   "track_id",
#   "track_name",
#   "track_artist",
#   "track_album_id",
#   "track_album_name",
#   "playlist_name",
#   "playlist_id",
#   "playlist_subgenre",
#   "language" ]
# df.drop(drop_cols,axis=1,inplace=True)
##################################################################

df.dtypes
# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car,num_but_cat = grab_col_names(df, cat_th=3, car_th=20)
num_cols = [col for col in num_cols if col not in ["track_popularity", "duration_ms"]]

# key threshold=5 olayından dolayı
# numerice gidiyordu ama o num görünümlü categoric
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

################################################################
# 3. BASE MODELS
################################################################


