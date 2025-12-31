# 🎧 MusicDNA

**MusicDNA**, Spotify şarkılarına ait ses özellikleri (audio features) ve şarkı sözlerinden çıkarılan duygu sinyallerini (NRCLex) kullanarak, **denetimsiz öğrenme (unsupervised learning)** yöntemleriyle kişilik temelli müzik dinleme eğilimlerini analiz etmeyi amaçlayan bir projedir.

Proje kapsamında **PCA + KMeans** yaklaşımı kullanılarak, şarkılar doğal kümelere ayrılmış ve bu kümeler **yorumlanabilir müzik kişilikleri** olarak anlamlandırılmıştır.

---

## 📌 Proje Genel Bakışı

MusicDNA, kullanıcının seçtiği sınırlı sayıdaki şarkı üzerinden:

- Spotify tarafından sağlanan **audio feature**’ları kullanır  
- Şarkı sözlerinden **NRCLex** ile duygu özellikleri çıkarır  
- **PCA** ile boyut indirgeme uygular  
- **KMeans** algoritması ile şarkıları benzerliklerine göre kümeler  
- Kullanıcının en yakın olduğu kümeye göre **yorumlanabilir bir müzik kişiliği profili** üretir  

Proje, müzik zevklerinin keskin sınırlar yerine **geçişli ve hibrit** bir yapıya sahip olduğu varsayımıyla tasarlanmıştır.

---

## 📊 Veri Seti

Bu projede Kaggle üzerinde herkese açık olarak paylaşılan aşağıdaki veri seti kullanılmıştır:

- **Audio Features and Lyrics of Spotify Songs**  
  https://www.kaggle.com/datasets/muhammad/audio-features-and-lyrics-of-spotify-songs

Veri seti aşağıdaki bileşenleri içermektedir:
- Spotify audio feature’ları  
- Şarkı sözleri (lyrics)  
- Genre ve subgenre bilgileri  
- Sanatçı, şarkı adı ve popülerlik gibi metadata alanları  

> ⚠️ **Önemli Not:**  
> `genre` ve `subgenre` değişkenleri **kümeleme işlemine dahil edilmemiştir**.  
> Bu alanlar yalnızca sonuçların **yorumlanması ve kümelerin adlandırılması** amacıyla kullanılmıştır.

---

## 🧩 Kullanılan Özellikler

Kümeleme işlemi iki ana özellik grubuna dayanmaktadır:

### 🎵 Audio Features
Spotify tarafından sağlanan sayısal ses özellikleri:
- danceability  
- energy  
- loudness  
- speechiness  
- acousticness  
- instrumentalness  
- liveness  
- valence  
- tempo  

### 📝 Lyrics Emotion Features
Şarkı sözlerinden **NRCLex** ile çıkarılan duygu skorları:
- joy  
- sadness  
- anger  
- fear  
- trust  
- anticipation  
- surprise  
- disgust  
- positive  
- negative  

Model karmaşıklığını azaltmak ve ayrıştırıcı gücü artırmak amacıyla, bu duygular arasından **en etkili olanlar** seçilerek kullanılmıştır.

---

## ⚙️ Yöntem

### 1️⃣ Ön İşleme
- Eksik ve geçersiz veriler temizlenir  
- Sayısal değişkenler **StandardScaler** ile ölçeklendirilir  

### 2️⃣ Şarkı Sözlerinden Duygu Çıkarımı
- NRCLex kullanılarak her şarkı için duygu skorları elde edilir  
- Duygu skorları, şarkı sözlerinin uzunluğuna bağlı yanlılığı azaltmak için normalize edilir  

### 3️⃣ Boyut İndirgeme
- **PCA** uygulanarak veri daha düşük boyutlu bir uzaya taşınır  
- Açıklanan varyans oranına göre bileşen sayısı belirlenir  

### 4️⃣ Kümeleme
- **KMeans** algoritması ile şarkılar benzerliklerine göre kümelenir  
- Küme sayısı **Elbow** ve **Silhouette** analizleri ile belirlenir  

### 5️⃣ Küme İsimlendirme
- Her kümenin ortalama özellikleri **z-score** yöntemiyle analiz edilir  
- Kümeler, baskın müzikal ve duygusal özelliklerine göre anlamlı isimlerle etiketlenir  

---

## 🧠 Proje Yapısı ve Veri Akışı

### Veri Akışı

1. **Veri Seti Yükleme**
   - Spotify şarkı verileri `spotify_songs.csv` dosyasından okunur  
   - Dosya yolu, farklı çalışma ortamlarına uyumlu olacak şekilde dinamik olarak tespit edilir  

2. **Özellik Çıkarımı**
   - Spotify audio feature’ları  
   - Lyrics üzerinden NRCLex duygu skorları  

3. **Ön İşleme**
   - Sayısal değişkenler standartlaştırılır  
   - Eksik ve geçersiz veriler temizlenir  

4. **Boyut İndirgeme**
   - PCA uygulanarak veri daha kompakt bir temsil uzayına taşınır  

5. **Kümeleme**
   - KMeans algoritması ile şarkılar kümelenir  

6. **Küme Yorumlama**
   - Her küme baskın özelliklerine göre analiz edilir ve isimlendirilir  

7. **Kullanıcı Analizi**
   - Kullanıcının seçtiği şarkılar üzerinden bir kullanıcı profili oluşturulur  
   - En yakın küme belirlenerek müzik kişiliği yorumu yapılır  

---

## ▶️ Projeyi Çalıştırma Sırası

Uygulama çalıştırılmadan önce modelin eğitilmesi, değerlendirilmesi ve gerekli artefaktların oluşturulması gerekir.  
Aşağıdaki adımlar **belirtilen sırayla** çalıştırılmalıdır.

### 1) Ana Pipeline (Model ve Artefakt OLuşturma)**
- python pipeline.py
**Ne yapılır?**
- Spotify veri seti (spotify_songs.csv) yüklenir
- Eksik ve geçersiz kayıtlar temizlenir
- Audio feature’lar ve lyrics tabanlı duygu özellikleri çıkarılır
- Tüm sayısal özellikler StandardScaler ile ölçeklendirilir
- PCA uygulanarak boyut indirgeme yapılır
- KMeans ile ana kümeleme gerçekleştirilir
- PCA bileşenleri (pca1, pca2) veri setine eklenir
- Küme etiketleri (cluster, cluster_name) oluşturulur

**Ne elde edilir?**
- musicdna_artifacts.joblib
- eğitilmiş scaler
- PCA modeli
- KMeans modeli
- nihai veri seti (df_final)
- kullanılan özellik listesi

---

### 2) Kümeleme Değerlendirmesi (Evaluation)**
- python evaluation/pca_distance_test.py
- python evaluation/silhouette_test.py
- python evaluation/genre_cluster_analysis.py

- Bu adımda, pipeline.py ile oluşturulan artefaktlar kullanılarak
  kümeleme kalitesi nicel olarak değerlendirilir.


**pca_distance_test.py**
**Ne yapar?**
- Her noktanın kendi kümesinin centroid’ine olan uzaklığını hesaplar
- En yakın diğer kümenin centroid’ine olan uzaklıkla karşılaştırır

**Ne elde edilir?**
- Küme içi vs küme dışı mesafe istatistikleri
- PCA mesafe histogramı (evaluation/outputs/)


**silhouette_test.py**
**Ne yapar?**
- PCA uzayında silhouette skorunu hesaplar
- Örnek bazlı silhouette dağılımını analiz eder

**Ne elde edilir?**
- Genel silhouette score
- Silhouette histogram görseli


**genre_cluster_analysis.py**
**Ne yapar?**
- Küme etiketleri ile genre dağılımını karşılaştırır
- Genre bilgisinin modele girdi olmadan nasıl dağıldığını gösterir

**Ne elde edilir?**
- Cluster × Genre yüzde tablosu (.csv)
- Kümelerin müzikal olarak anlamlı olduğunu gösteren destekleyici analiz


### 3) Raporlama ve Görselleştirme (Reports)
**python reports/cluster_evaluation_metrics.py**
**python reports/cluster_evaluation_visuals.py**

- Bu adım, raporda kullanılan nihai kalite metriklerini ve ana görselleri üretir.

## cluster_evaluation_metrics.py
**Ne yapar?**
- Silhouette Score
- Inertia (SSD)
- Küme içi / küme dışı mesafe özetlerini hesaplar

**Ne elde edilir?**
- Kümeleme kalitesini sayısal olarak özetleyen çıktılar
- Raporun nicel değerlendirme bölümü için temel metrikler


## cluster_evaluation_visuals.py

**Ne yapar?**
- PCA 2D küme dağılım grafiği
- Centroid işaretleri
- Silhouette plot
- Centroid distance heatmap

**Ne elde edilir?**
- Raporun ana figürleri
  (cluster map, silhouette plot, distance matrix)

### 4) Alt Kümeleme (Subclustering – Opsiyonel)
## python subcluster_cluster0.py
- Bu adım yalnızca belirli bir kümenin (Cluster 0) kendi iç yapısını
  daha ayrıntılı analiz etmek amacıyla kullanılır.

**Ne yapar?**
- Ana clustering sonucunda elde edilen Cluster 0’ı izole eder
- Bu küme üzerinde yeniden KMeans uygular
- Alt müzikal yapıların (ör. Pop / Rap) ayrışmasını inceler

**Ne elde edilir?**
- Subcluster etiketleri
- PCA tabanlı alt küme görselleri
- Küme isimlendirmesini destekleyen ek analizler
- Bu adım ana modeli değiştirmez, yalnızca yorumlama derinliği sağlar.

### 5) Streamlit Uygulaması (Son Adım)
## streamlit run app.py
- Bu adımda, daha önce oluşturulmuş artefaktlar kullanılarak
  kullanıcıya yönelik interaktif arayüz çalıştırılır.

## Ne yapar?
**musicdna_artifacts.joblib** dosyasını yükler
- Kullanıcıdan şarkı girdileri alır
- Kullanıcının müzik kişiliğini analiz eder
- En yakın kümeyi ve önerileri görsel olarak sunar

**Ne elde edilir?**
- Çalışan MusicDNA web arayüzü

