import streamlit as st
from Utils import show_footer

st.set_page_config(
    page_title="Pertanyaan Umum",
    page_icon="🔍",
)

st.markdown("<h1 style='text-align: center;'>Seputar Hasil Panen dan Ekspor Kakao di Indonesia</h1>", unsafe_allow_html=True)

with st.expander("Apa tujuan dari website ini?"):
    st.write("Website ini dirancang untuk melakukan pengelompokan (clustering) dan pemetaan lokasi terkait hasil panen dan ekspor kakao di Indonesia. Dengan website ini, pengguna dapat melihat sebaran wilayah berdasarkan tingkat produksi, luas areal, produktivitas, volume dan nilai kakao, sehingga dapat digunakan untuk analisis pola distribusi maupun potensi ekspor secara visual dan terstruktur.")

with st.expander("Data apa saja yang digunakan dalam website ini?"):
    st.write("Website ini menggunakan data mengenai hasil panen kakao per provinsi dan kabupaten di Indonesia, yang mencakup luas areal (ha), produksi (ton), dan produktivitas (kg/ha). Selain itu, website ini juga menggunakan data ekspor kakao, yang terdiri dari volume ekspor (ton) dan nilai ekspor (USD). Data geografis berupa koordinat lokasi wilayah penghasil kakao juga digunakan untuk keperluan pemetaan visual.")

with st.expander("Bagaimana cara website ini melakukan pengelompokkan (clustering)?"):
    st.write("Website ini menggunakan beberapa algoritma clustering, yaitu KMeans, KMedoids, dan Bisecting KMeans, untuk mengelompokkan wilayah penghasil kakao di Indonesia. Pengelompokan dilakukan berdasarkan data yang tersedia, seperti volume hasil panen, luas areal, produktivitas, volume ekspor, dan nilai ekspor, sehingga dapat membantu dalam menganalisis pola produksi dan distribusi kakao antar wilayah.")

with st.expander("Apa manfaat dari clustering hasil panen kakao?"):
    st.write("Dengan melakukan clustering, pengguna dapat melihat pola distribusi dan karakteristik wilayah penghasil kakao yang serupa. Hal ini membantu dalam merencanakan strategi distribusi dan pemasaran.")

with st.expander("Bagaimana cara menggunakan website ini?"):
    st.write("Pengguna cukup mengakses website dan memilih provinsi, kabupaten atau wilayah yang ingin dianalisis. website akan menampilkan hasil clustering dan memberikan wawasan mengenai hasil panen dan ekspor kakao di wilayah tersebut.")

with st.expander("Bagaimana cara menggunakan website ini?"):
    st.write("Pengguna cukup mengakses website dan memilih provinsi, kabupaten, atau data ekspor yang ingin dianalisis. Website akan menampilkan hasil clustering dan pemetaan serta memberikan visualisasi lainnya mengenai data panen dan ekspor kakao di wilayah tersebut.")

with st.expander("Apakah bisa mengubah data yang digunakan dalam website?"):
    st.write("Saat ini, program menggunakan data yang telah disediakan, namun tetap dapat mengganti atau memperbarui data tersebut dengan mengikuti format template yang tersedia. Template data dapat diunduh pada halaman Panen dan Ekspor. Pastikan data yang diunggah sesuai dengan struktur yang ditentukan agar dapat diproses dengan benar oleh program.")

with st.expander("Bagaimana cara menghubungi developer jika menemui masalah?"):
    st.write("Jika mengalami masalah atau memiliki pertanyaan lebih lanjut, dapat menghubungi developer melalui email yang tertera di halaman about website.")

# Footer
show_footer()