import streamlit as st
from Utils import show_footer

st.set_page_config(
    page_title="Tentang",
    page_icon="👨‍💻",
    layout='wide',
)

# Custom CSS dengan spacing yang lebih ketat
st.markdown("""
<style>
    .profile-img {
        border-radius: 50%;
        border: 3px solid #4F8BF9;
        margin-bottom: 0.5rem;
    }
    .section-title {
        color: #4F8BF9;
        border-bottom: 2px solid #4F8BF9;
        margin-top: 1rem !important;
    }
    .compact-text {
        line-height: 1.4;
        margin-bottom: 0.5rem;
        text-align: justify;
    }
    .compact-list {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section - lebih compact
col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.image("./assets/profil.png", width=200, use_container_width=False, 
            caption="Jefri Jaya")

with col2:
    st.title('Jefri Jaya', anchor=False)
    st.markdown("""
    <div class="compact-text">
    📚 Mahasiswa Teknik Informatika di Universitas Tarumanagara<br>
    🔍 Memiliki minat di bidang Machine Learning & Web Programming
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="compact-text">
    Halo! Saya Jefri Jaya, saat ini sedang menyelesaikan studi sarjana. Saya senang belajar hal-hal baru dan selalu berusaha berkembang setiap hari. Saya suka membaca buku tentang semi filsafat dan pengembangan diri. Saya percaya bahwa kerja keras dan rasa ingin tahu adalah kunci untuk mencapai tujuan, dan setiap kegagalan adalah langkah penting menuju keberhasilan.
            
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="compact-text">
    Dosen Pembimbing 1:<br>
    Teny Handhyani, S.Kom., M.Kom., Ph.D. <br>
    Dosen Pembimbing 2: <br>
    Janson Hendryli, S.Kom., M.Kom.
    </div>
    """, unsafe_allow_html=True)

# Divider dengan margin lebih kecil
st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# Main Content dengan spacing lebih ketat
tab1, tab2, tab3 = st.tabs(["Kemampuan", "Pengalaman", "Hobi"])

with tab1:
    st.markdown('<h2 class="section-title">Kemampuan</h2>', unsafe_allow_html=True)
    
    cols = st.columns(1)
    st.markdown("**Programming & Data**")
    st.markdown("""
    <div class="compact-list">
    - Python<br>
    - Laravel<br>
    - React JS <br>
    - SQL<br>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="section-title">Pengalaman</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="compact-text" style="margin-top: 0.5rem;">
    <strong>🌐 Web Developer Intern</strong><br>
    <em>Maritim Muda Nusantara 2024</em><br>
    • Developed Laravel & React Js website<br>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<h2 class="section-title">Hobi</h2>', unsafe_allow_html=True)
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Hobi & Aktivitas**")
        st.markdown("""
        <div class="compact-list">
        - 🏃‍♂️ Jogging<br>
        - 🎱 Billiard<br>
        - ✈️ Jalan-jalan<br>
        - ♟️ Chess Strategy
        </div>
        """, unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown("**Favorite Reads**")
        st.markdown("""
        <div class="compact-list">
        - Filosofi Teras<br>
        - Berani Tidak Disukai <br>
        - The Psychology of Money <br>
        - Zero To One<br>
        - The Alchemist
        </div>
        """, unsafe_allow_html=True)


# Footer
show_footer()