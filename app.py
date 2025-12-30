# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import ARTIFACT_PATH
from artifacts import load_artifacts
from engine import music_dna_engine
from persona import seed_from_user_songs, generate_dynamic_comment

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="MusicDNA",
    page_icon="🎧",
    layout="wide",
)

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

# ----------------------------
# Header
# ----------------------------
st.title("🎧 MusicDNA")
st.caption("Dinlediğin şarkılar seni anlatır. MusicDNA ile bu hikâyeyi keşfet 🎶")

# ----------------------------
# Load artifacts
# ----------------------------
try:
    art = get_artifacts()
except Exception as e:
    st.error("❌ Artifacts okunamadı. Önce terminalde `python pipeline.py` çalıştırıp joblib üretmelisin.")
    st.exception(e)
    st.stop()

df_final = art["df_final"]
scaler = art["scaler"]
pca = art["pca"]
kmeans = art["kmeans"]
model_features = art["model_features"]
cluster_names = art["cluster_names"]

song_options = build_song_options(df_final)

# ----------------------------
# Sidebar – Input
# ----------------------------
with st.sidebar:
    st.header("🧾 Girdi")
    st.caption("💡 2–4 şarkı seçmek genelde daha stabil sonuç verir.")

    n = st.number_input("Kaç şarkı gireceksin?", min_value=1, max_value=10, value=3, step=1)
    st.divider()

    user_song_list = []
    for i in range(int(n)):
        st.subheader(f"🎵 Şarkı {i+1}")

        mode = st.radio(
            "Giriş yöntemi",
            ["Listeden seç", "Manuel gir"],
            key=f"mode_{i}",
            horizontal=True,
        )

        if mode == "Listeden seç":
            sel = st.selectbox("Şarkı seç", song_options, key=f"sel_{i}")
            name, artist = sel.split(" — ", 1)
            user_song_list.append({"name": name, "artist": artist})
        else:
            name = st.text_input("Şarkı adı", key=f"name_{i}", placeholder="Örn: Blinding Lights")
            artist = st.text_input("Sanatçı (opsiyonel)", key=f"artist_{i}", placeholder="Örn: The Weeknd")
            user_song_list.append({"name": name, "artist": artist if artist else None})

        st.divider()

    top_n = st.slider("🎯 Kaç öneri gösterilsin?", min_value=3, max_value=10, value=5, step=1)

    st.caption("🧠 Seçtiğin şarkılara göre MusicDNA profilini çıkarır.")
    run = st.button("🧬 Analizi Başlat", type="primary", use_container_width=True)

# ----------------------------
# Run analysis
# ----------------------------
if run:
    # temizle
    user_song_list = [
        s for s in user_song_list
        if isinstance(s.get("name"), str) and s["name"].strip() != ""
    ]

    if len(user_song_list) == 0:
        st.warning("⚠️ En az 1 şarkı girmelisin.")
        st.stop()

    with st.spinner("🔎 Analiz yapılıyor... (profil + yakınlık + öneriler)"):
        dna, cluster_id, soft_top2, recs, used = music_dna_engine(
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
        st.error("❌ Şarkı bulunamadı. Listeden seçmeyi deneyebilir veya yazımı kontrol edebilirsin.")
        st.stop()

    # dna iki satırlı gelebilir: "Başlık\nAlt başlık"
    title, subtitle = (str(dna).split("\n", 1) + [""])[:2]

    seed = seed_from_user_songs(user_song_list)
    comment = generate_dynamic_comment(title, seed=seed, max_sentences=4)

    left, right = st.columns([1, 1])

    # ----------------------------
    # LEFT: Profile + soft assignment + comment + used
    # ----------------------------
    with left:
        st.success(f"✨ **Müzikal Kişiliğin:** {title}")
        if subtitle.strip():
            st.caption(f"📌 {subtitle}")

        # -----------------------------
        # Yakınlık yorumunu daha akıllı yaz (hibrit sadece yakınsa)
        # -----------------------------
        if soft_top2 and len(soft_top2) >= 2:
            p1 = float(soft_top2[0]["prob"]) * 100
            p2 = float(soft_top2[1]["prob"]) * 100
            diff = p1 - p2

            name1 = str(soft_top2[0]["cluster_name"]).split("\n", 1)[0]
            name2 = str(soft_top2[1]["cluster_name"]).split("\n", 1)[0]

            HYBRID_DIFF_TH = 10.0  # hibrit eşiği (yüzde puan)

            if diff < HYBRID_DIFF_TH:
                st.warning(
                    f"🟡 **Hibrit profil:** "
                    f"**{name1}** (%{p1:.1f}) ile "
                    f"**{name2}** (%{p2:.1f}) birbirine yakın görünüyor. "
                    f"Daha stabil sonuç için 1–2 şarkı daha ekleyebilirsin."
                )
            else:
                st.info(
                    f"🟢 **Baskın profil:** **{name1}** (%{p1:.1f}) açık ara önde.\n\n"
                    f"🟠  **İkinci yakın profil:** **{name2}** (%{p2:.1f})"
                )

        # Soft assignment listesi
        if soft_top2:
            st.subheader("🎯 Yakınlıklar")
            for item in soft_top2:
                pct = round(float(item["prob"]) * 100, 1)
                cname = str(item["cluster_name"]).split("\n", 1)[0]
                st.write(f"• {cname}: **%{pct}**")

        st.subheader("🔮 Kısa Yorum")
        st.info(comment)

        st.subheader("🔗 Eşleşen Şarkılar")
        st.dataframe(pd.DataFrame(used), use_container_width=True, hide_index=True)

    # ----------------------------
    # RIGHT: Genre proximity + recommendations
    # ----------------------------
    with right:
        st.subheader("📊 Bu DNA hangi türlere yakın?")

        if "playlist_genre" in df_final.columns and cluster_id is not None:
            g = (
                df_final[df_final["cluster"] == cluster_id]["playlist_genre"]
                .value_counts(normalize=True)
                .head(6)
                .reset_index()
            )
            g.columns = ["genre", "ratio"]
            g["ratio"] = (g["ratio"] * 100).round(1)

            st.dataframe(g, use_container_width=True, hide_index=True)
            st.bar_chart(g.set_index("genre")["ratio"])
        else:
            st.info("ℹ️ Genre bilgisi (playlist_genre) bulunamadı, bu panel gösterilemiyor.")

        st.subheader("🧬 DNA'na Uygun Öneriler")

        if recs is None or len(recs) == 0:
            st.warning("⚠️ Bu cluster için öneri bulunamadı.")
        else:
            cols = ["track_name", "track_artist", "track_popularity"]
            if "playlist_genre" in recs.columns:
                cols.insert(2, "playlist_genre")

            st.dataframe(
                recs[cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
