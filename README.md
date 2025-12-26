# MusicDNA 🎧

MusicDNA, Spotify şarkılarına ait ses özellikleri ve şarkı sözlerinden çıkarılan
duygu sinyallerini (NRCLex) kullanarak, denetimsiz öğrenme yöntemleri
(PCA + KMeans) ile kişilik temelli müzik dinleme eğilimlerini analiz etmeyi amaçlar.

## Proje Genel Bakışı (Project Overview)
MusicDNA, kullanıcının seçtiği küçük bir şarkı listesi üzerinden:

- Spotify audio feature’larını ve şarkı sözlerinden çıkarılan duygu özelliklerini kullanır
- PCA ile boyut indirgeme uygular
- KMeans algoritması ile şarkıları kümeler
- Kullanıcının en yakın olduğu kümeye göre yorumlanabilir bir
  **“müzik kişiliği (Music Personality)”** özeti üretir

## Veri Seti (Dataset)
Bu projede, Kaggle üzerinde herkese açık olarak paylaşılan aşağıdaki veri seti kullanılmıştır:

- **Audio Features and Lyrics of Spotify Songs**
  - https://www.kaggle.com/datasets/muhammad/audio-features-and-lyrics-of-spotify-songs

Veri seti; Spotify audio feature’ları, şarkı sözleri, genre / subgenre bilgileri
ve diğer metadata alanlarını içermektedir.

## Kullanılan Özellikler
Kümeleme işlemi iki ana katmandan elde edilen özellikler kullanılarak yapılmıştır:

- **Audio Features**  
  Spotify tarafından sağlanan sayısal ses özellikleri
  (danceability, energy, valence vb.)

- **Lyrics Emotion Features**  
  Şarkı sözlerinden **NRCLex** ile çıkarılan duygu skorları
  (joy, sadness, anger, fear, trust vb.)

> **Not:** `genre` ve `subgenre` değişkenleri,
> **kümeleme işlemine dahil edilmemiştir**.
> Bu alanlar yalnızca sonuçların yorumlanması ve kümelerin adlandırılması amacıyla kullanılmıştır.

## Yöntem (Method)
1. **Önişleme**
   - Eksik verilerin ele alınması
   - Sayısal değişkenlerin ölçeklendirilmesi (StandardScaler)

2. **Şarkı Sözlerinden Duygu Çıkarımı**
   - NRCLex kullanılarak her şarkı için duygu skorları elde edilir

3. **Boyut İndirgeme**
   - PCA uygulanarak veri daha düşük boyutlu bir uzaya taşınır
   - Açıklanan varyans oranına göre bileşen sayısı belirlenir

4. **Kümeleme**
   - KMeans algoritması ile şarkılar benzerliklerine göre kümelenir
   - Küme sayısı, elbow ve silhouette analizleri ile belirlenmiştir

5. **Küme İsimlendirme**
   - Her kümenin ortalama özellikleri z-skor yöntemi ile karşılaştırılır
   - Kümeler, baskın müzikal ve duygusal özelliklerine göre anlamlı isimlerle etiketlenir

## Proje Yapısı

Bu projede amaç, kullanıcının seçtiği şarkılar üzerinden
müzik dinleme eğilimlerini analiz ederek
kişilik temelli bir **MusicDNA profili** oluşturmaktır.

### Veri Akışı
1. **Veri Seti Yükleme**
   - Spotify şarkı verileri `spotify_songs.csv` dosyasından okunur
   - Dosya yolu, farklı çalışma ortamlarına uyumlu olacak şekilde dinamik olarak tespit edilir

2. **Özellik Çıkarımı**
   - Spotify audio feature’ları
   - Şarkı sözlerinden NRCLex ile çıkarılan duygu skorları

3. **Önişleme**
   - Sayısal değişkenler standartlaştırılır
   - Eksik veya geçersiz veriler temizlenir

4. **Boyut İndirgeme**
   - PCA uygulanarak veri daha düşük boyutlu hale getirilir

5. **Kümeleme**
   - KMeans algoritması ile şarkılar kümelenir

6. **Küme Yorumlama**
   - Her küme, baskın özelliklerine göre analiz edilir ve isimlendirilir

7. **Kullanıcı Analizi**
   - Kullanıcının seçtiği şarkılar üzerinden bir kullanıcı profili oluşturulur
   - En yakın küme belirlenerek müzik kişiliği yorumu yapılır

### Önemli Notlar
- Proje tamamen **denetimsiz öğrenme (unsupervised learning)** yaklaşımı ile geliştirilmiştir
- Genre ve subgenre bilgileri, modele girdi olarak verilmemiştir



