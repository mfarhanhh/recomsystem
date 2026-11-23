"""
Aplikasi Streamlit: Association Rule dan Recommender System (Book-Crossing)

Jalankan dengan:
    streamlit run app.py

Data yang dibutuhkan (letakkan di folder yang sama dengan app.py, di folder `data/`,
atau upload lewat sidebar):
    - BX-Books.csv
    - BX-Book-Ratings.csv
    - BX-Users.csv  (opsional, hanya untuk ringkasan data)
"""

import io
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Association Rule & Recommender Buku",
    page_icon="📚",
    layout="wide",
)

NAMA_FILE = {
    "buku": "BX-Books.csv",
    "user": "BX-Users.csv",
    "rating": "BX-Book-Ratings.csv",
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def cari_sumber(upload, nama_file):
    """Prioritas: file upload -> folder saat ini -> folder data/. Return bytes / path / None."""
    if upload is not None:
        return upload.getvalue()
    for kandidat in (nama_file, os.path.join("data", nama_file)):
        if os.path.exists(kandidat):
            return kandidat
    return None


def _buka(sumber):
    return io.BytesIO(sumber) if isinstance(sumber, (bytes, bytearray)) else sumber


@st.cache_data(show_spinner="Membaca data buku...")
def muat_buku(sumber):
    df = pd.read_csv(
        _buka(sumber), sep=";", encoding="latin-1", escapechar="\\", low_memory=False
    )
    df = df[["ISBN", "Book-Title", "Book-Author"]]
    df.columns = ["isbn", "judul", "penulis"]
    return df


@st.cache_data(show_spinner="Membaca data user...")
def muat_user(sumber):
    return pd.read_csv(_buka(sumber), sep=";", encoding="latin-1", na_values="NULL")


@st.cache_data(show_spinner="Membaca data rating...")
def muat_rating(sumber):
    df = pd.read_csv(_buka(sumber), sep=";", encoding="latin-1")
    df.columns = ["user", "isbn", "rating"]
    return df


# =============================================================================
# 2. ASSOCIATION RULE
# =============================================================================
@st.cache_data(show_spinner=False)
def buat_basket(rating, min_buku_user, jumlah_buku_top):
    """Satu user = satu transaksi, buku yang pernah dirating = item."""
    jml_user = rating.groupby("user").size()
    data = rating[rating["user"].isin(jml_user[jml_user >= min_buku_user].index)]

    top_buku = data["isbn"].value_counts().head(jumlah_buku_top).index
    data = data[data["isbn"].isin(top_buku)]

    # buang user yang bukunya < 2 (tidak bisa membentuk aturan)
    jml_user = data.groupby("user").size()
    data = data[data["user"].isin(jml_user[jml_user >= 2].index)]

    return pd.crosstab(data["user"], data["isbn"]).astype(bool)


@st.cache_data(show_spinner="Menjalankan Apriori...")
def cari_rules(basket, min_support, min_confidence):
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return frequent, pd.DataFrame()

    frequent["ukuran"] = frequent["itemsets"].apply(len)

    try:
        rules = association_rules(
            frequent, metric="confidence", min_threshold=min_confidence
        )
    except TypeError:
        # mlxtend versi baru mewajibkan num_itemsets
        rules = association_rules(
            frequent,
            num_itemsets=len(basket),
            metric="confidence",
            min_threshold=min_confidence,
        )
    except ValueError:
        rules = pd.DataFrame()
    return frequent, rules


# =============================================================================
# 3. RECOMMENDER SYSTEM
# =============================================================================
@st.cache_resource(show_spinner="Menyiapkan matriks rating...")
def siapkan_model(_rating, sig, min_rating_user, min_rating_buku, test_size, seed):
    """Filter rating eksplisit, split train/test, lalu bentuk matriks user x buku."""
    eks = _rating[_rating["rating"] > 0]

    jml_user = eks.groupby("user").size()
    eks = eks[eks["user"].isin(jml_user[jml_user >= min_rating_user].index)]

    jml_buku = eks.groupby("isbn").size()
    eks = eks[eks["isbn"].isin(jml_buku[jml_buku >= min_rating_buku].index)]

    jml_user = eks.groupby("user").size()
    eks = eks[eks["user"].isin(jml_user[jml_user >= 5].index)]

    if len(eks) < 50:
        return None

    train, test = train_test_split(eks, test_size=test_size, random_state=seed)

    list_user = sorted(eks["user"].unique())
    list_buku = sorted(eks["isbn"].unique())
    idx_user = {u: i for i, u in enumerate(list_user)}
    idx_buku = {b: i for i, b in enumerate(list_buku)}

    # matriks user x buku dari data train (0 = belum dirating)
    R = np.zeros((len(list_user), len(list_buku)))
    for u, b, r in train.itertuples(index=False):
        R[idx_user[u], idx_buku[b]] = r

    # kemiripan antar buku (cosine)
    sim_buku = cosine_similarity(R.T)
    np.fill_diagonal(sim_buku, 0)

    # baseline
    rata_global = train["rating"].mean()
    jumlah = (R > 0).sum(axis=0)
    rata_buku = np.where(jumlah > 0, R.sum(axis=0) / np.maximum(jumlah, 1), rata_global)

    return {
        "eks": eks,
        "train": train,
        "test": test,
        "list_user": list_user,
        "list_buku": list_buku,
        "idx_user": idx_user,
        "idx_buku": idx_buku,
        "R": R,
        "sim_buku": sim_buku,
        "rata_global": rata_global,
        "rata_buku": rata_buku,
    }


def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))


def mae(y, p):
    return np.mean(np.abs(y - p))


@st.cache_data(show_spinner="Menghitung evaluasi model (bisa beberapa saat)...")
def evaluasi(_m, kunci, k):
    """Bandingkan baseline, item-based CF, dan user-based CF pada data test."""
    R = _m["R"]
    sim_buku = _m["sim_buku"]
    rata_buku = _m["rata_buku"]
    test = _m["test"]

    y_true = test["rating"].values
    u_test = test["user"].map(_m["idx_user"]).values
    b_test = test["isbn"].map(_m["idx_buku"]).values

    pred_global = np.full(len(y_true), _m["rata_global"])
    pred_rata_buku = rata_buku[b_test]

    # --- Item-based CF ---
    pred_item = []
    for u, b in zip(u_test, b_test):
        dinilai = np.nonzero(R[u])[0]  # buku yang sudah dirating user u
        sim = sim_buku[b, dinilai]
        if len(sim) > k:  # ambil k buku paling mirip
            urut = np.argsort(sim)[-k:]
            dinilai = dinilai[urut]
            sim = sim[urut]
        if sim.sum() > 0:
            pred_item.append((sim * R[u, dinilai]).sum() / sim.sum())
        else:
            pred_item.append(rata_buku[b])
    pred_item = np.array(pred_item)

    # --- User-based CF ---
    sim_user = cosine_similarity(R)
    np.fill_diagonal(sim_user, 0)

    pred_user = []
    for u, b in zip(u_test, b_test):
        pemberi = np.nonzero(R[:, b])[0]  # user lain yang sudah merating buku b
        sim = sim_user[u, pemberi]
        if len(sim) > k:
            urut = np.argsort(sim)[-k:]
            pemberi = pemberi[urut]
            sim = sim[urut]
        if sim.sum() > 0:
            pred_user.append((sim * R[pemberi, b]).sum() / sim.sum())
        else:
            pred_user.append(rata_buku[b])
    pred_user = np.array(pred_user)

    nama = ["Rata-rata global", "Rata-rata buku", "Item-based CF", "User-based CF"]
    preds = [pred_global, pred_rata_buku, pred_item, pred_user]
    return pd.DataFrame(
        {
            "Metode": nama,
            "RMSE": [rmse(y_true, p) for p in preds],
            "MAE": [mae(y_true, p) for p in preds],
        }
    ).round(3)


# =============================================================================
# SIDEBAR: SUMBER DATA & PARAMETER
# =============================================================================
st.title("📚 Association Rule & Recommender System")
st.caption("Dataset Book-Crossing (BX-Books, BX-Users, BX-Book-Ratings)")

with st.sidebar:
    st.header("📁 Sumber Data")
    st.caption(
        "Jika file sudah ada di folder aplikasi (atau folder `data/`), "
        "upload tidak diperlukan."
    )
    up_buku = st.file_uploader("BX-Books.csv", type="csv")
    up_rating = st.file_uploader("BX-Book-Ratings.csv", type="csv")
    up_user = st.file_uploader("BX-Users.csv (opsional)", type="csv")

    st.header("⚙️ Parameter")

    with st.expander("Association Rule", expanded=True):
        min_buku_user = st.number_input(
            "Minimal buku per user", 2, 50, 5, help="User dengan buku lebih sedikit dibuang."
        )
        jumlah_buku_top = st.slider("Jumlah buku terpopuler", 50, 500, 300, step=50)
        min_support = st.slider("Minimum support", 0.005, 0.10, 0.01, step=0.005, format="%.3f")
        st.caption("⚠️ Support yang terlalu kecil bisa membuat Apriori lambat dan boros memori.")
        min_confidence = st.slider("Minimum confidence", 0.1, 1.0, 0.3, step=0.05)

    with st.expander("Recommender System", expanded=True):
        min_rating_user = st.number_input("Minimal rating per user", 5, 100, 10)
        min_rating_buku = st.number_input("Minimal rating per buku", 5, 200, 20)
        test_size = st.slider("Porsi data test", 0.1, 0.5, 0.2, step=0.05)
        k = st.slider("Jumlah tetangga (k)", 5, 100, 30, step=5)
        seed = st.number_input("Random state", 0, 9999, 42)

# =============================================================================
# LOAD DATA
# =============================================================================
src_buku = cari_sumber(up_buku, NAMA_FILE["buku"])
src_rating = cari_sumber(up_rating, NAMA_FILE["rating"])
src_user = cari_sumber(up_user, NAMA_FILE["user"])

if src_buku is None or src_rating is None:
    st.info(
        "Data belum ditemukan. Upload **BX-Books.csv** dan **BX-Book-Ratings.csv** "
        "lewat sidebar, atau letakkan keduanya di folder yang sama dengan `app.py`."
    )
    st.stop()

try:
    buku = muat_buku(src_buku)
    rating = muat_rating(src_rating)
    user = muat_user(src_user) if src_user is not None else None
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

info_buku = buku.drop_duplicates("isbn", keep="last").set_index("isbn")
judul = info_buku["judul"].to_dict()
penulis = info_buku["penulis"].to_dict()

tab_data, tab_ar, tab_eval, tab_rek = st.tabs(
    ["📊 Ringkasan Data", "🔗 Association Rule", "⭐ Evaluasi Recommender", "📖 Rekomendasi"]
)

# =============================================================================
# TAB 1: RINGKASAN DATA
# =============================================================================
with tab_data:
    st.subheader("1. Ringkasan Data")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah buku", f"{buku.shape[0]:,}")
    c2.metric("Jumlah user", f"{user.shape[0]:,}" if user is not None else "-")
    c3.metric("Jumlah rating", f"{rating.shape[0]:,}")

    st.markdown("**Contoh data rating**")
    st.dataframe(rating.head(), hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Missing value di data buku**")
        st.dataframe(buku.isna().sum().rename("jumlah missing").to_frame())
        if user is not None and "Age" in user.columns:
            st.write(f"Missing umur user: **{user['Age'].isna().mean() * 100:.1f}%**")

    with col_b:
        st.markdown("**Distribusi rating**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        rating["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Distribusi rating")
        ax.set_xlabel("rating")
        ax.set_ylabel("jumlah")
        st.pyplot(fig)
        plt.close(fig)

    n0 = (rating["rating"] == 0).sum()
    st.info(
        f"Rating 0 berjumlah {n0:,} dari {len(rating):,} "
        f"({n0 / len(rating) * 100:.1f}%). Rating ini berarti user berinteraksi dengan "
        "buku tetapi tidak memberi nilai (implisit). Rating eksplisit adalah 1-10."
    )

# =============================================================================
# TAB 2: ASSOCIATION RULE
# =============================================================================
with tab_ar:
    st.subheader("2. Association Rule")
    st.write(
        "Satu user dianggap satu transaksi, dan buku yang pernah dirating "
        "(termasuk rating 0) dianggap item di dalam transaksi itu. Supaya tidak terlalu "
        f"berat, dipakai {jumlah_buku_top} buku yang paling banyak dirating."
    )

    if st.button("▶️ Jalankan Association Rule", type="primary"):
        st.session_state["jalan_ar"] = True

    if st.session_state.get("jalan_ar"):
        basket = buat_basket(rating, int(min_buku_user), int(jumlah_buku_top))

        if basket.shape[0] == 0:
            st.warning("Tidak ada transaksi yang memenuhi syarat. Longgarkan parameter.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Jumlah transaksi (user)", f"{basket.shape[0]:,}")
            m2.metric("Jumlah item (buku)", f"{basket.shape[1]:,}")
            m3.metric("Rata-rata buku per user", f"{basket.sum(axis=1).mean():.1f}")

            frequent, rules = cari_rules(basket, min_support, min_confidence)

            st.markdown(f"**Jumlah frequent itemset: {len(frequent):,}**")
            if len(frequent) > 0:
                st.dataframe(
                    frequent["ukuran"]
                    .value_counts()
                    .sort_index()
                    .rename_axis("ukuran itemset")
                    .rename("jumlah")
                    .to_frame()
                )

            if rules.empty:
                st.warning(
                    "Tidak ada rules yang ditemukan. Coba turunkan minimum support "
                    "atau minimum confidence."
                )
            else:
                st.markdown(f"**Jumlah rules: {len(rules):,}**")

                tampil = rules.copy()
                for kolom in ("antecedents", "consequents"):
                    tampil[kolom] = tampil[kolom].apply(
                        lambda x: ", ".join(judul.get(i, i) for i in x)
                    )
                hasil = tampil[["antecedents", "consequents", "support", "confidence", "lift"]]

                cc1, cc2 = st.columns(2)
                urut_by = cc1.selectbox("Urutkan berdasarkan", ["lift", "confidence", "support"])
                n_tampil = cc2.slider("Jumlah baris ditampilkan", 5, 100, 10, step=5)

                hasil = hasil.sort_values(urut_by, ascending=False).round(3)
                st.dataframe(hasil.head(n_tampil), hide_index=True)

                st.download_button(
                    "⬇️ Unduh semua rules (CSV)",
                    hasil.to_csv(index=False).encode("utf-8"),
                    file_name="association_rules.csv",
                    mime="text/csv",
                )

                fig, ax = plt.subplots(figsize=(7, 4))
                sc = ax.scatter(
                    rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis"
                )
                fig.colorbar(sc, ax=ax, label="lift")
                ax.set_xlabel("support")
                ax.set_ylabel("confidence")
                ax.set_title("Support vs confidence")
                st.pyplot(fig)
                plt.close(fig)

                st.info(
                    "Rules dengan lift tinggi biasanya berisi buku dalam satu seri. Nilai "
                    "support cenderung kecil karena jumlah buku banyak dan setiap user hanya "
                    "merating sedikit buku."
                )
    else:
        st.caption("Atur parameter di sidebar, lalu klik tombol di atas.")

# =============================================================================
# SIAPKAN MODEL RECOMMENDER (dipakai tab 3 dan 4)
# =============================================================================
sig = (len(rating), int(rating["rating"].sum()))
model = siapkan_model(
    rating,
    sig,
    int(min_rating_user),
    int(min_rating_buku),
    float(test_size),
    int(seed),
)

# =============================================================================
# TAB 3: EVALUASI RECOMMENDER
# =============================================================================
with tab_eval:
    st.subheader("3. Recommender System (Collaborative Filtering)")
    st.write(
        "Hanya rating eksplisit (1-10) yang dipakai, karena rating 0 berarti user tidak "
        "memberi nilai. Data difilter supaya matriksnya tidak terlalu kosong."
    )

    if model is None:
        st.warning("Data setelah difilter terlalu sedikit. Longgarkan parameter di sidebar.")
    else:
        eks, train, test, R = model["eks"], model["train"], model["test"], model["R"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah rating", f"{len(eks):,}")
        m2.metric("Jumlah user", f"{eks['user'].nunique():,}")
        m3.metric("Jumlah buku", f"{eks['isbn'].nunique():,}")
        m4.metric("Rata-rata rating", f"{eks['rating'].mean():.2f}")

        persen_isi = (R > 0).mean() * 100
        st.write(
            f"Split data **{int((1 - test_size) * 100)}:{int(test_size * 100)}** "
            f"(train {len(train):,} / test {len(test):,}). "
            f"Ukuran matriks user x buku: **{R.shape[0]:,} x {R.shape[1]:,}**, "
            f"sel terisi: **{persen_isi:.2f}%**."
        )

        if st.button("▶️ Jalankan Evaluasi", type="primary"):
            st.session_state["jalan_eval"] = True

        if st.session_state.get("jalan_eval"):
            kunci = (sig, int(min_rating_user), int(min_rating_buku), float(test_size), int(seed))
            tabel = evaluasi(model, kunci, int(k))

            st.markdown("**Perbandingan metode**")
            st.dataframe(tabel, hide_index=True)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            xpos = np.arange(len(tabel))
            ax.bar(xpos - 0.2, tabel["RMSE"], width=0.4, label="RMSE")
            ax.bar(xpos + 0.2, tabel["MAE"], width=0.4, label="MAE")
            ax.set_xticks(xpos)
            ax.set_xticklabels(tabel["Metode"], rotation=15)
            ax.set_ylabel("error (lebih kecil lebih baik)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            terbaik = tabel.loc[tabel["RMSE"].idxmin()]
            baseline = tabel.loc[tabel["Metode"] == "Rata-rata buku", "RMSE"].iloc[0]
            st.success(
                f"Metode terbaik berdasarkan RMSE: **{terbaik['Metode']}** "
                f"(RMSE {terbaik['RMSE']}, MAE {terbaik['MAE']}). "
                f"Baseline rata-rata buku memiliki RMSE {baseline}."
            )
            st.caption(
                "Jika matriks sangat kosong, sulit mencari user yang benar-benar mirip, "
                "sehingga selisih antar metode biasanya kecil dan jangan dibaca terlalu jauh."
            )
        else:
            st.caption("Klik tombol di atas untuk menghitung RMSE dan MAE tiap metode.")

# =============================================================================
# TAB 4: REKOMENDASI
# =============================================================================
with tab_rek:
    st.subheader("4. Contoh Rekomendasi")

    if model is None:
        st.warning("Data setelah difilter terlalu sedikit. Longgarkan parameter di sidebar.")
    else:
        R = model["R"]
        sim_buku = model["sim_buku"]
        list_user = model["list_user"]
        list_buku = model["list_buku"]

        # ---------- Rekomendasi untuk user ----------
        st.markdown("### Rekomendasi untuk user (item-based CF)")
        default_u = int((R > 0).sum(axis=1).argmax())  # user dengan rating terbanyak

        cu1, cu2, cu3 = st.columns([2, 1, 1])
        pilih_user = cu1.selectbox(
            "Pilih user",
            options=list(range(len(list_user))),
            index=default_u,
            format_func=lambda i: f"User {list_user[i]} ({int((R[i] > 0).sum())} rating)",
        )
        batas_suka = cu2.slider("Dianggap suka jika rating ≥", 1, 10, 9)
        top_n = cu3.slider("Jumlah rekomendasi", 5, 30, 10)

        u = pilih_user
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown(f"**Buku yang disukai (rating ≥ {batas_suka})**")
            suka = np.nonzero(R[u] >= batas_suka)[0]
            if len(suka) == 0:
                st.write("User ini tidak punya buku dengan rating setinggi itu.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "judul": [judul.get(list_buku[j], list_buku[j]) for j in suka],
                            "penulis": [penulis.get(list_buku[j], "-") for j in suka],
                            "rating": [int(R[u, j]) for j in suka],
                        }
                    ),
                    hide_index=True,
                )

        with col_r:
            st.markdown(f"**Top {top_n} rekomendasi**")
            # skor tiap buku = rata-rata rating buku mirip yang sudah dia nilai
            skor = sim_buku @ R[u] / (sim_buku @ (R[u] > 0) + 1e-9)
            skor[R[u] > 0] = 0  # buku yang sudah dirating dibuang
            top = [j for j in np.argsort(skor)[::-1][:top_n] if skor[j] > 0]
            if not top:
                st.write("Tidak ada rekomendasi yang bisa dihitung untuk user ini.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "judul": [judul.get(list_buku[j], list_buku[j]) for j in top],
                            "penulis": [penulis.get(list_buku[j], "-") for j in top],
                            "skor": [round(float(skor[j]), 2) for j in top],
                        }
                    ),
                    hide_index=True,
                )

        st.caption(
            "Rekomendasi dihitung dari matriks rating data train. Jika skor antar buku "
            "berdekatan, artinya rekomendasi kurang spesifik karena datanya jarang (sparse)."
        )

        # ---------- Buku yang mirip ----------
        st.divider()
        st.markdown("### Buku yang mirip")

        populer = np.argsort((R > 0).sum(axis=0))[::-1]  # urut dari yang paling banyak dirating
        cb1, cb2 = st.columns([3, 1])
        pilih_buku = cb1.selectbox(
            "Pilih buku (diurutkan dari yang paling banyak dirating)",
            options=[int(j) for j in populer],
            format_func=lambda j: (
                f"{judul.get(list_buku[j], list_buku[j])} - "
                f"{penulis.get(list_buku[j], '-')} ({int((R[:, j] > 0).sum())} rating)"
            ),
        )
        n_mirip = cb2.slider("Jumlah buku mirip", 3, 20, 5)

        j = pilih_buku
        mirip = np.argsort(sim_buku[j])[::-1][:n_mirip]
        st.dataframe(
            pd.DataFrame(
                {
                    "judul": [judul.get(list_buku[m], list_buku[m]) for m in mirip],
                    "penulis": [penulis.get(list_buku[m], "-") for m in mirip],
                    "similarity": [round(float(sim_buku[j, m]), 3) for m in mirip],
                }
            ),
            hide_index=True,
        )
        st.caption("Similarity yang rendah menandakan kemiripan antar buku belum kuat.")
