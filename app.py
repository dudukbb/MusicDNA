# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import ARTIFACT_PATH
from artifacts import load_artifacts
from engine import music_dna_engine
from persona import seed_from_user_songs, generate_dynamic_comment

st.set_page_config(page_title="MusicDNA", page_icon="🎧", layout="wide")

@st.cache_resource
def get_artifacts():
    return load_artifacts(ARTIFACT_PATH)

@st.cache_data
def build_song_options(df: pd.DataFrame):
    opts = (
        df[["track_name", "track_artist"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["track_name", "track_artist"])
    )
    return [f"{r.track_name} — {r.track_artist}" for r in opts.itertuples(index=False)]

st.title("🎧 MusicDNA – Müzikal Kişilik Analizi")
st.caption("Şarkı seç → MusicDNA profili + fal vari yorum + öneriler")

try:
    art = get_artifacts()
except Exception as e:
    st.error("Artifacts okunamadı. Önce: python pipeline.py çalıştırıp joblib üretmelisin.")
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
        mode = st.radio("Giriş yöntemi", ["Listeden seç", "Manuel gir"], key=f"mode_{i}", horizontal=True)

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
        top_n=top_n,
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

            # playlist_genre varsa göster (yoksa eski haliyle göster)
            cols = ["track_name", "track_artist", "track_popularity"]
            if "playlist_genre" in recs.columns:
                cols.insert(2, "playlist_genre")  # artist'ten sonra genre gelsin

            st.dataframe(
                recs[cols].reset_index(drop=True),
                use_container_width=True
            )
