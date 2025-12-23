import hashlib
import random
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MusicDNA", page_icon="🎧", layout="wide")

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("musicdna_artifacts.joblib")

# ----------------------------
# Helpers
# ----------------------------
def _normalize(s: str) -> str:
    return str(s).lower().strip()

def find_song_row(df: pd.DataFrame, track_name: str, track_artist: str | None = None):
    if not track_name or not isinstance(track_name, str) or track_name.strip() == "":
        return None

    tn = _normalize(track_name)
    mask = df["track_name"].str.lower().str.contains(tn, na=False, regex=False)
    hits = df[mask].copy()
    if hits.empty:
        return None

    if track_artist and isinstance(track_artist, str) and track_artist.strip() != "":
        ta = _normalize(track_artist)
        artist_mask = hits["track_artist"].str.lower().str.contains(ta, na=False, regex=False)
        if artist_mask.any():
            hits = hits[artist_mask]

    return hits.sort_values("track_popularity", ascending=False).iloc[0]

def music_dna_engine(
    user_song_list,
    df_final,
    model_features,
    scaler,
    pca,
    kmeans,
    cluster_names,
    top_n=5,
):
    found_rows = []
    for s in user_song_list:
        row = find_song_row(df_final, s.get("name"), s.get("artist"))
        if row is not None:
            found_rows.append(row)

    if not found_rows:
        return None, None, []

    user_vec = pd.DataFrame(found_rows)[model_features].mean().to_frame().T
    user_scaled = scaler.transform(user_vec)
    user_pca = pca.transform(user_scaled)
    cluster_id = int(kmeans.predict(user_pca)[0])

    dna_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    used_names = {r["track_name"].lower() for r in found_rows if isinstance(r.get("track_name"), str)}
    recs = df_final[df_final["cluster"] == cluster_id].copy()
    recs = recs[~recs["track_name"].str.lower().isin(used_names)]
    top_recs = recs.sort_values("track_popularity", ascending=False).head(top_n)

    used_pretty = [{"track_name": r["track_name"], "track_artist": r["track_artist"]} for r in found_rows]
    return dna_name, top_recs[["track_name", "track_artist", "track_popularity"]], used_pretty

@st.cache_data
def build_song_options(df: pd.DataFrame):
    opts = (
        df[["track_name", "track_artist"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["track_name", "track_artist"])
    )
    return [f"{r.track_name} — {r.track_artist}" for r in opts.itertuples(index=False)]

# ----------------------------
# “Fal” yorum sistemi (spotify.py ile aynı)
# ----------------------------
PERSONA_TEXT = {
    "Mutlu & Coşkulu": [
        "Sen müziği sadece dinlemiyorsun; enerji topluyorsun.",
        "Pozitif tınılar sende anında mod yükseltir.",
        "Neşeli parçalar seni hızla motive eden bir düğme gibi."
    ],
    "Agresif & Dinamik": [
        "Müzikte güç ve sertlik seni canlı tutuyor.",
        "Enerji yükseldikçe sen daha net hissediyorsun.",
        "Sert ritimler sende bir ‘gaz’ etkisi yaratıyor."
    ],
    "Hüzünlü & Melankolik": [
        "Müzik senin için yüzleşme alanı gibi; duygudan kaçmıyorsun.",
        "Derin ve duygusal parçalar sende daha uzun kalıyor.",
        "Herkes ritme kapılırken sen sözlere takılan taraftasın."
    ],
    "Huzurlu & Sakin": [
        "Müzik sende gürültüyü azaltan bir filtre gibi çalışıyor.",
        "Sakin tınılar senin denge alanın.",
        "Akış ve dinginlik, zevkinin gizli anahtarı."
    ],
    "Dans/Party": [
        "Ritmi yakaladığında ayakta durmak zor.",
        "Beat iyiyse gerisi teferruat.",
        "Playlist’in ‘kıpır kıpır’ çalışıyor."
    ],
    "Enstrümantal/Odak": [
        "Sözlerden çok melodilerle bağ kuruyorsun.",
        "Müzik sende bazen ‘odak modu’ açıyor.",
        "Enstrümanlar konuşsun, sen dinlersin."
    ],
    "Sözel/Rap": [
        "Sen ritim kadar mesajı da takip ediyorsun.",
        "Anlatısı olan parçalar seni daha çabuk yakalıyor.",
        "Söz ağırlığı sende bir ‘hikâye’ hissi bırakıyor."
    ],
    "Akustik": [
        "Doğal tınılar sana daha gerçek geliyor.",
        "Akustik taraf sende ‘samimiyet’ arayışı gibi.",
        "Yapay olmayan sesler sende daha çok iz bırakıyor."
    ],
}

ENDING_BY_MOOD = {
    "Mutlu & Coşkulu": [
        "Bu enerji seni kolay kolay aşağı çekmez.",
        "Pozitiflik senin temel frekansın.",
        "Mod yükselten tarafın oldukça baskın."
    ],
    "Agresif & Dinamik": [
        "Güçlü hissettiğinde müzik seninle aynı yönde akıyor.",
        "Bu sertlik sende net bir duruş yaratıyor."
    ],
    "Hüzünlü & Melankolik": [
        "Bu derinlik herkeste bulunmaz.",
        "Duygularla bu kadar yakın olmak güçlü bir taraf."
    ],
    "Huzurlu & Sakin": [
        "Bu sakinlik senin doğal alanın.",
        "Denge arayışın müzikte çok net hissediliyor."
    ],
}

GENERAL_ENDINGS = [
    "Bu müzik zevki sana özgü bir imza gibi.",
    "Müzik tercihlerin rastgele değil, oldukça tutarlı.",
    "Bu dinleme tarzı seni anlatmanın sessiz bir yolu.",
    "Playlist’inde bile bir karakter var."
]

def seed_from_user_songs(user_song_list) -> int:
    key = "|".join([f"{s.get('name','')}-{s.get('artist','')}" for s in user_song_list]).lower()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def generate_dynamic_comment(cluster_label: str, seed: int | None = None, max_sentences=4) -> str:
    rng = random.Random(seed)
    parts = [p.strip() for p in cluster_label.split("/")] if cluster_label else []
    sentences = []

    for p in parts:
        if p in PERSONA_TEXT:
            sentences.append(rng.choice(PERSONA_TEXT[p]))

    rng.shuffle(sentences)
    sentences = sentences[:max(1, max_sentences - 1)]

    main_mood = parts[0] if parts else None
    if main_mood in ENDING_BY_MOOD:
        sentences.append(rng.choice(ENDING_BY_MOOD[main_mood]))
    else:
        sentences.append(rng.choice(GENERAL_ENDINGS))

    return "\n".join(sentences)

# ----------------------------
# UI
# ----------------------------
st.title("🎧 MusicDNA – Müzikal Kişilik Analizi")
st.caption("Şarkı seç → MusicDNA profili + fal vari yorum + öneriler")

try:
    art = load_artifacts()
except Exception as e:
    st.error("musicdna_artifacts.joblib okunamadı. Önce spotify.py çalıştırıp export üretmelisin.")
    st.exception(e)
    st.stop()

df_final = art["df_final"]
scaler = art["scaler"]
pca = art["pca"]
kmeans = art["kmeans"]
model_features = art["model_features"]
cluster_names = art["cluster_names"]

song_options = build_song_options(df_final)

with st.sidebar:
    st.subheader("Girdi")
    st.caption("💡 2-3 şarkı seçersen profil daha stabil olur.")
    n = st.number_input("Kaç şarkı gireceksin?", min_value=1, max_value=10, value=3, step=1)

    user_song_list = []
    for i in range(int(n)):
        st.markdown(f"### Şarkı {i+1}")

        mode = st.radio(
            "Giriş yöntemi",
            ["Listeden seç", "Manuel gir"],
            key=f"mode_{i}",
            horizontal=True
        )

        if mode == "Listeden seç":
            sel = st.selectbox("Şarkı seç", song_options, key=f"sel_{i}")
            name, artist = sel.split(" — ", 1)
            user_song_list.append({"name": name, "artist": artist})
        else:
            name = st.text_input("Şarkı adı", key=f"name_{i}")
            artist = st.text_input("Sanatçı (opsiyonel)", key=f"artist_{i}")
            user_song_list.append({"name": name, "artist": artist if artist else None})

    top_n = st.slider("Kaç öneri gösterilsin?", min_value=3, max_value=10, value=5, step=1)
    run = st.button("Analiz Et ✅", type="primary")

if run:
    user_song_list = [s for s in user_song_list if isinstance(s.get("name"), str) and s["name"].strip() != ""]

    dna, recs, used = music_dna_engine(
        user_song_list=user_song_list,
        df_final=df_final,
        model_features=model_features,
        scaler=scaler,
        pca=pca,
        kmeans=kmeans,
        cluster_names=cluster_names,
        top_n=top_n
    )

    st.divider()

    if dna is None:
        st.error("Şarkı bulunamadı. Listeden seçmeyi deneyebilir veya yazımı kontrol edebilirsin.")
    else:
        seed = seed_from_user_songs(user_song_list)
        comment = generate_dynamic_comment(dna, seed=seed, max_sentences=4)

        left, right = st.columns([1, 1])

        with left:
            st.success(f"**Müzikal Kişiliğin:** {dna}")
            st.write("**🔮 Kısa yorum:**")
            st.info(comment)
            st.write("**Eşleşen şarkılar:**")
            st.dataframe(pd.DataFrame(used), use_container_width=True)

        with right:
            st.write("**DNA'na uygun öneriler:**")
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)
