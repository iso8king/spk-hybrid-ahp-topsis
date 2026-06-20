import streamlit as st 
import numpy as np
import pandas as pd

def hitung_bobot_ahp(matriks):
    n = len(matriks)

    #normalisasi
    jumlah_kolom = matriks.sum(axis=0);
    matriks_normal = matriks/jumlah_kolom

    # hitung weight
    weight = matriks_normal.mean(axis=1)

    # eigen value dan cv
    aw = matriks @ weight
    lambda_ = np.mean(aw / weight)

    # ci
    ci = (lambda_ - n) / n - 1
    ri = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    cr = ci/ri.get(n, 1.49) if n > 2 else 0.0

    return weight, cr

def hitung_topsis(alternatif, weight, tipe_kriteria):
     """
    Menghitung perankingan alternatif dengan metode TOPSIS.
    
    Parameters:
    data_alternatif: matrix m x n (m alternatif, n kriteria)
    bobot: array bobot kriteria dari AHP
    tipe_kriteria: array 'benefit' atau 'cost' untuk setiap kriteria
    
    Returns:
    df_hasil: DataFrame dengan skor dan ranking
    """
     data = np.array(alternatif, dtype=float)
     m, n = data.shape
     normalisasi = np.sqrt((data ** 2).sum(axis=0))
     matriks_normalisasi = data / normalisasi

     # matriks normalisasi terbobot
     matriks_terbobot = matriks_normalisasi * weight

     # solusi ideal positif dan negatif
     ideal_positif = []
     ideal_negatif = []

     for j in range(n):
        if tipe_kriteria[j] == 'benefit':
               ideal_positif.append(matriks_terbobot[:,j].max())
               ideal_negatif.append(matriks_terbobot[:, j].min())
        else :
             ideal_positif.append(matriks_terbobot[:, j].min())
             ideal_negatif.append(matriks_terbobot[:, j].max())
     
     # jarak solusi ideal
     jarak_positif = np.sqrt(((matriks_terbobot - ideal_positif) ** 2).sum(axis=1))
     jarak_negatif = np.sqrt(((matriks_terbobot - ideal_negatif) ** 2).sum(axis=1))

     # skor preferensi
     skor = jarak_negatif / (jarak_positif + jarak_negatif)

     rank = np.argsort(skor)[::-1] + 1 
     return pd.DataFrame({
        'Alternatif': nama_alternatif,
        'Skor': skor,
        'Ranking': rank
     }).sort_values('Ranking')


st.set_page_config(page_title="SPK Hybrid AHP-TOPSIS", layout="wide")

st.title("Sistem Pendukung Keputusan Hybrid")
st.subheader("Metode AHP + TOPSIS")

with st.sidebar:
     st.header("Input Data")
     st.subheader("Kriteria")
     jumlah_kriteria = st.number_input('Jumlah Kriteria', min_value=2, value=3)
     nama_kriteria = []
     for i in range(jumlah_kriteria):
          nama = st.text_input(f'Nama Kriteria {i+1}', value=f'K{i+1}', key=f'criteria_name_k{i+1}')
          nama_kriteria.append(nama)
     st.divider()

     st.subheader('Alternatif')
     jumlah_alternatif = st.number_input('Jumlah Alternatif', min_value=2, value=3)
     nama_alternatif = []
     for i in range(jumlah_alternatif):
          nama = st.text_input(f'Nama Alternatif {i+1}', value=f'A{i+1}', key=f'alternative_name_A{i+1}')
          nama_alternatif.append(nama)

tab1, tab2, tab3 = st.tabs(["Matriks Perbandingan AHP", 'Data Alternatif' , 'Hasil'])

with tab1:
     st.header("Matriks Perbandingan Berpasangan (AHP)")
     st.caption("Isi matriks dengan skala Saaty (1-9). Nilai >1 berarti kriteria baris lebih penting dari kolom.")
     st.image("https://image2.slideserve.com/5065675/slide22-l.jpg", caption='Saaty Scale', use_container_width=True)
     matriks_ahp = np.ones((jumlah_kriteria, jumlah_kriteria))
     cols = st.columns(jumlah_kriteria)
     
     for i in range(jumlah_kriteria):
          for j in range(jumlah_kriteria):
               if i == j:
                    matriks_ahp[i][j] = 1.0
               elif i < j:
                    value = st.number_input(f'{nama_kriteria[i]} vs {nama_kriteria[j]}', min_value=1.0, value=1.0, step=1.0, max_value = 9.0, key=f'ahp_{i}_{j}')
                    matriks_ahp[i][j] = value
                    matriks_ahp[j][i] = 1.0/ value
     df_matriks = pd.DataFrame(matriks_ahp, index = nama_kriteria, columns=nama_alternatif)
     st.dataframe(df_matriks, use_container_width=True)

     if(st.button("Hitung Bobot AHP", key='btn_ahp')):
          weight, cr = hitung_bobot_ahp(matriks_ahp)
          st.session_state['weight'] = weight
          st.session_state['cr'] = cr
          st.success(f"Consistency Value : {cr:.3f} {'(Konsisten)' if cr <= 0.1 else '(Tidak Konsiten)'}")
          st.info(f"**Bobot Kriteria:** {', '.join([f'{nama}: {b:.3f}' for nama, b in zip(nama_kriteria, weight)])}")

with tab2:
     st.header("Data Nilai Alternatif per kriteria")
     st.caption("Isi nilai untuk setiap alternatif pada setiap kriteria.")
     data_alt = np.zeros((jumlah_alternatif, jumlah_alternatif))

     st.subheader("Tipe Kriteria")
     tipe = []
     cols_tipe = st.columns(jumlah_kriteria)
     for i, nama in enumerate(nama_kriteria):
          with cols_tipe[i]:
               tipe.append(st.selectbox(f"{nama}", options=['benefit' , 'cost'], key=f'tipe_{i}'))
     st.subheader("Nilai Alternatif")
     for i in range(jumlah_alternatif):
          cols = st.columns(jumlah_kriteria)
          for j in range(jumlah_kriteria):
               with cols[j]:
                    data_alt[i][j] = st.number_input(f"{nama_alternatif[i]} - {nama_kriteria[j]}", min_value=0.0, value=float(i+1)*10 + j*5, step=1.0, key=f'alt_{i}_{j}')
     df_data = pd.DataFrame(data_alt, index=nama_alternatif, columns=nama_kriteria)
     st.dataframe(df_data, use_container_width=True)
    
     st.session_state['data_alternatif'] = data_alt
     st.session_state['tipe_kriteria'] = tipe

with tab3:
     st.header("Hasil Ranking")
     
     if 'weight' in st.session_state and 'data_alternatif' in st.session_state:
          weight = st.session_state['weight']
          data = st.session_state['data_alternatif']
          tipe = st.session_state['tipe_kriteria']

          st.subheader("Weight Kriteria dari AHP")
          df_weight = pd.DataFrame({'Kriteria' : nama_kriteria, "Weight" : weight})
          st.dataframe(df_weight, use_container_width=True)

          st.subheader("Tipe Kriteria")
          df_tipe = pd.DataFrame({"Kriteria" : nama_kriteria, "Tipe" : tipe})
          st.dataframe(df_tipe, use_container_width=True)

          if st.button("Hitung Ranking" , key='btn_topsis'):
               hasil = hitung_topsis(data, weight, tipe)

               st.subheader("Hasil")
               st.dataframe(hasil, use_container_width=True)

               st.subheader("Visualisasi Skor")
               st.bar_chart(hasil.set_index('Alternatif')['Skor'])

               winner = hasil.iloc[0]['Alternatif']
               st.success(f'Alternatif Terbaik : {winner}')

               with st.expander("🔍 Detail Perhitungan TOPSIS"):
                st.write("Rumus dan langkah-langkah perhitungan TOPSIS:")
                st.markdown("""
                1. **Normalisasi Matriks**: M = X / √(ΣX²)
                2. **Normalisasi Terbobot**: V = M × bobot
                3. **Solusi Ideal Positif (A+)**: nilai maksimum untuk benefit, minimum untuk cost
                4. **Solusi Ideal Negatif (A-)**: nilai minimum untuk benefit, maksimum untuk cost
                5. **Jarak Euclidean**: D+ = √(Σ(V - A+)²), D- = √(Σ(V - A-)²)
                6. **Skor Preferensi**: S = D- / (D+ + D-)
                """)
          else:
               st.warning("Belum Ada Data, Silahkan Hitung Bobot AHP terlebih dahulu")
               