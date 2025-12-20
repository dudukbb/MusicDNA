import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="MusicDNA", page_icon="🎧", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load("musicdna_artifacts.joblib")

art = load_artifacts()
df_final = art["df_final"]
scaler = art["scaler"]
pca = art["pca"]
kmeans = art["kmeans"]
model_features = art["model_features"]
cluster_names = art["cluster_names"]

# --- senin yardımcı fonksiyonların (minimum gerekli olanlar) ---
def _normalize(s: str) -> str:
    return str(s).lower().strip()

def find_song_row(df, track_name, track_artist=None):
    if not track_name:
        return None
    tn = _normalize(track_name)
    mask = df["track_name"].str.lower().str.contains(tn, na=False, regex=False)
    hits = df[mask].copy()
    if hits.empty:
        return None
    if track_artist:
        ta = _normalize(track_artist)
        artist_mask = hits["track_artist"].str.lower().str.contains(ta, na=False, regex=False)
        if artist_mask.any():
            hits = hits[artist_mask]
    return hits.sort_values("track_popularity", ascending=False).iloc[0]

def music_dna_engine(user_song_list, df=df_final, top_n=5):
    found_rows = []
    for s in user_song_list:
        row = find_song_row(df, s.get("name"), s.get("artist"))
        if row is not None:
            found_rows.append(row)

    if not found_rows:
        return None, None, []

    user_vec = pd.DataFrame(found_rows)[model_features].mean().to_frame().T
    user_scaled = scaler.transform(user_vec)
    user_pca = pca.transform(user_scaled)  # PCA'lı modelle uyumlu
    cluster_id = int(kmeans.predict(user_pca)[0])

    dna_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    used_names = {r["track_name"].lower() for r in found_rows if isinstance(r.get("track_name"), str)}
    recs = df[df["cluster"] == cluster_id].copy()
    recs = recs[~recs["track_name"].str.lower().isin(used_names)]
    top_recs = recs.sort_values("track_popularity", ascending=False).head(top_n)

    used_pretty = [{"track_name": r["track_name"], "track_artist": r["track_artist"]} for r in found_rows]
    return dna_name, top_recs[["track_name", "track_artist", "track_popularity"]], used_pretty


# ---------------- UI ----------------
st.title("🎧 MusicDNA – Müzikal Kişilik Analizi")
st.caption("Şarkı adı / sanatçı gir → MusicDNA profili + öneriler")

with st.sidebar:
    st.subheader("Girdi")
    n = st.number_input("Kaç şarkı gireceksin?", min_value=1, max_value=10, value=2, step=1)

    user_song_list = []
    for i in range(int(n)):
        st.markdown(f"**Şarkı {i+1}**")
        name = st.text_input("Şarkı adı", key=f"name_{i}")
        artist = st.text_input("Sanatçı (opsiyonel)", key=f"artist_{i}")
        user_song_list.append({"name": name, "artist": artist if artist else None})

    run = st.button("Analiz Et ✅", type="primary")

if run:
    dna, recs, used = music_dna_engine(user_song_list, top_n=5)

    if dna is None:
        st.error("Şarkı bulunamadı. Yazımı kontrol edip tekrar deneyin.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.success(f"**Müzikal Kişiliğin:** {dna}")
            st.write("**Eşleşen şarkılar:**")
            st.dataframe(pd.DataFrame(used), use_container_width=True)

        with col2:
            st.write("**DNA'na uygun öneriler:**")
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)
