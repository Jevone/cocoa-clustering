import streamlit as st
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn_extra.cluster import KMedoids
from io import BytesIO
import base64

# 1: KONFIGURASI PANEN & EKSPOR
REQUIRED_PANEN = ['Lokasi', 'Latitude', 'Longitude']
FEATURE_PANEN = ['Luas', 'Produksi', 'Produktivitas']
REQUIRED_EKSPOR = ['Negara Tujuan', 'Latitude', 'Longitude']
FEATURE_EKSPOR = ['Volume', 'Nilai']

CLUSTER_LABELS = {
    2: ['Rendah', 'Tinggi'],
    3: ['Rendah', 'Sedang', 'Tinggi'],
    4: ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi'],
    5: ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
    6: ['Sangat Rendah', 'Rendah', 'Cukup Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
    7: ['Sangat Rendah', 'Rendah', 'Cukup Rendah', 'Sedang', 'Cukup Tinggi', 'Tinggi', 'Sangat Tinggi']
}

COLOR_PALETTE = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'black']

def translate_error_message(raw_message):
    if "Found array with 0 sample(s)" in raw_message:
        return "Data tidak tersedia atau kosong. Silakan unggah file dengan isi yang sesuai."
    return raw_message

def initialize_clustering_model(method, n_clusters):
    models = {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42, max_iter=100, init='k-means++'),
        'KMedoids': KMedoids(n_clusters=n_clusters, random_state=60, max_iter=100, init='k-medoids++'),
        'Bisecting KMeans': BisectingKMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
    }
    return models.get(method)

def get_cluster_labels(n_clusters):
    return CLUSTER_LABELS.get(n_clusters, [f"Cluster {i}" for i in range(n_clusters)])

def get_image_download_link(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'''
    <a href="data:image/png;base64,{b64}" 
    download="{filename}.png"
    style="
        display: inline-block;
        padding: 0.3em 1em;
        background: none;
        color: #ffffff ;
        border-radius: 5px;
        text-decoration: none;
        font-weight: none;
    ">
    🖼️ Download PNG
    </a>
    '''
    return href

def get_fig_buffer(fig, dpi=300):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=dpi)
    buffer.seek(0)
    return buffer

def show_footer():
    st.markdown("""
    <hr style='margin: 0.5rem 0;'>
    <div style='text-align: center; font-size: 0.9rem; padding: 0.5rem 0;'>
        © 2025 Jefri Jaya
    </div>
    """, unsafe_allow_html=True)
