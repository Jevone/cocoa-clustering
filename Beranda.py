import streamlit as st 
from Utils import show_footer

st.set_page_config(
    page_title="Beranda",
    page_icon="🍫",
)

# Judul halaman rata tengah
st.markdown("<h1 style='text-align: center;'>Perkebunan Kakao di Indonesia</h1>", unsafe_allow_html=True)

Image_kakao = "./assets/kakao.png"

col1, col2, col3 = st.columns([1, 6, 1])  # Kolom tengah lebih lebar
with col2:
    st.image(Image_kakao, caption="Buah Kakao", width=500)  # Sesuaikan width


# Paragraf dengan teks rata kiri-kanan (justify)
st.markdown("""
    <div style='text-align: justify; text-indent: 40px;'>
    Kakao merupakan salah satu komoditas unggulan Indonesia di sektor perkebunan yang memiliki peran strategis dalam menunjang perekonomian nasional. Keunggulan agroklimat Indonesia menjadikan negara ini sebagai wilayah yang sangat potensial untuk pengembangan tanaman kakao secara berkelanjutan. Komoditas ini tidak hanya berperan sebagai sumber mata pencaharian utama bagi jutaan petani di berbagai daerah, tetapi juga berkontribusi signifikan terhadap penerimaan devisa negara melalui aktivitas ekspor. Oleh karena itu, kakao memiliki posisi penting dalam mendorong pertumbuhan ekonomi daerah dan memperkuat ketahanan ekonomi nasional, khususnya di sektor pertanian dan industri hilir pengolahan kakao.
    </div>
    """, unsafe_allow_html=True)

# Spasi antar paragraf
st.markdown("<br>", unsafe_allow_html=True)


# Paragraf dengan teks rata kiri-kanan (justify)
st.markdown("""
    <div style='text-align: justify; text-indent: 40px;'>
    Penelitian ini bertujuan untuk mengidentifikasi dan menganalisis pola distribusi hasil panen kakao di berbagai provinsi di Indonesia dengan menggunakan pendekatan berbasis data. Salah satu metode yang digunakan adalah clustering, yang memungkinkan dilakukannya pemetaan wilayah berdasarkan kesamaan karakteristik panen dan ekspor. Dengan pemetaan yang komprehensif ini, diharapkan dapat diketahui wilayah-wilayah yang memiliki potensi tinggi maupun rendah, sehingga kebijakan pembangunan sektor kakao dapat disesuaikan secara lebih tepat sasaran. Informasi ini sangat penting untuk mendukung perumusan strategi pengembangan komoditas kakao yang lebih efektif, adil, dan berorientasi pada keberlanjutan jangka panjang.
    </div>
    """, unsafe_allow_html=True)


with st.expander("🍫 Tahapan Mengolah Kakao menjadi Cokelat Batangan"):
    image1 = "./assets/kakao.png"
    image2 = "./assets/fermentasi.png"
    image3 = "./assets/penjemuran_kakao.png"
    image4 = "./assets/sangrai.png"
    image5 = "./assets/pembersihan.png"
    image6 = "./assets/penghalusan.png"
    image7 = "./assets/pencetakan.png"
    image8 = "./assets/pengemasan.png"

    st.write("1. Pengambilan Biji Kakao")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image1, caption="Buah Kakao", width=200)
    st.write("Buah kakao yang telah matang dibelah dan bijinya dikeluarkan.")

    st.write("2. Fermentasi")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image2, caption="Fermentasi Kakao", width=200)
    st.write("Biji kakao dimasukkan ke dalam alat fermentor dan difermentasi selama 7 hari.")

    st.write("3. Penjemuran")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image3, caption="Penjemuran Biji Kakao", width=200)
    st.write("Setelah fermentasi, biji kakao dijemur di bawah sinar matahari hingga kering")

    st.write("4. Penyangraian (Roasting)")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image4, caption="Penyangraian Kakao", width=200)
    st.write("Biji kakao kering disangrai di dalam kuali hingga benar-benar kering dan mengeluarkan aroma kakao..")

    st.write("5. Pembersihan")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image5, caption="Pembersihan Kakao", width=200)
    st.write("Biji kakao dibersihkan dari kulit ari dan kotoran lainnya.")

    st.write("6. Pencampuran Bahan")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image6, caption="Penghalusan Kakao", width=200)
    st.write("Biji kakao dimasukkan ke dalam blender, lalu ditambahkan minyak goreng, susu, dan gula dengan perbandingan 1:1:1:1. dan campuran diblender hingga halus dan membentuk pasta cokelat.")

    st.write("7. Pencetakan")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image7, caption="Pencetakan Kakao", width=200)
    st.write("Pasta cokelat dituangkan ke dalam cetakan dan dibekukan dalam freezer.")

    st.write("8. Pengemasan")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image8, caption="Pengemasan Kakao", width=200)
    st.write("Setelah mengeras, cokelat batang dikemas menggunakan aluminium foil dan dibungkus dengan kertas pembungkus cokelat.")

# Footer
show_footer()