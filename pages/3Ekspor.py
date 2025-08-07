import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import re
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import folium
from streamlit_folium import folium_static
from Utils import show_footer, initialize_clustering_model, get_cluster_labels,get_image_download_link, translate_error_message, get_fig_buffer, COLOR_PALETTE, REQUIRED_EKSPOR, FEATURE_EKSPOR

st.set_page_config(
    page_title="Ekspor",
    page_icon="🌍",
)

# ==============================================
# FUNGSI UTAMA
# ==============================================
def perform_clustering(uploaded_file, method, n_clusters):
    result = {'dataframe': None, 'metrics': None, 'error': None}
    
    try:
        # Proses validation data
        df = validate_input_data(uploaded_file)
        
        # Preprocessing
        features = df.drop(columns=REQUIRED_EKSPOR)
        x_scaled = StandardScaler().fit_transform(features)
        
        # Model training
        model = initialize_clustering_model(method, n_clusters)
        start_time = time.time()
        model.fit(x_scaled)
        labels = model.predict(x_scaled) if hasattr(model, 'predict') else model.labels_
        
        # Evaluation metrics
        silhouette = round(silhouette_score(x_scaled, labels), 4)
        dbi = round(davies_bouldin_score(x_scaled, labels), 4)
        
        # pemanggilan hasil metrics
        df['Cluster'] = labels
        result['dataframe'] = df
        result['metrics'] = {
            'processing_time': round(time.time() - start_time, 2),
            'silhouette': silhouette,
            'dbi': dbi,
            'method': method,
            'n_clusters': n_clusters
        }
        result['x_scaled'] = x_scaled
        
    # Error Handling
    except Exception as e:
        result['error'] = {
            'message': str(e),
            'traceback': traceback.format_exc()
        }
    return result

def validate_input_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    
    # Check required ekspor columns
    missing_cols = [col for col in REQUIRED_EKSPOR if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom: {missing_cols} tidak ditemukan. Pastikan file Anda memiliki kolom {missing_cols}")
    
    # Convert non-numeric columns
    numeric_cols = df.drop(columns=REQUIRED_EKSPOR).select_dtypes(include=['number']).columns
    non_numeric_cols = set(df.columns) - set(numeric_cols) - set(REQUIRED_EKSPOR)
    
    if non_numeric_cols:
        df[list(non_numeric_cols)] = df[list(non_numeric_cols)].apply(pd.to_numeric, errors='coerce')
        if df.isnull().any().any():
            error_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f" Nilai non-numerik ditemukan di kolom: {error_cols}")
    return df

def categorize_clusters(df):
    # Otomatis ambil kolom yang cocok dengan prefix
    feature_cols = [col for col in df.columns if any(x.lower() in col.lower() for x in ['volume', 'nilai'])]

    # Hitung rata-rata per cluster untuk setiap fitur
    cluster_means = df.groupby('Cluster')[feature_cols].mean()

    # Hitung skor agregat (rata-rata semua fitur per cluster)
    cluster_means['Skor_Agregat'] = cluster_means.mean(axis=1)

    # Urutkan skor agregat dari rendah ke tinggi
    cluster_means = cluster_means.sort_values(by='Skor_Agregat').reset_index()

    # Dapatkan label sesuai jumlah cluster
    labels = get_cluster_labels(len(cluster_means))

    # Mapping cluster ke label
    cluster_label_map = {
        row['Cluster']: labels[i]
        for i, row in cluster_means.iterrows()
    }

    # Tambahkan kolom Kategori ke df
    df['Kategori'] = df['Cluster'].map(cluster_label_map)

    return df

# ==============================================
# FUNGSI VISUALISASI
# ==============================================
def display_cluster_map(df):
    st.subheader("🗺️ Pemetaan Klasterisasi")
    st.write("Keterangan Cluster:")
    
    # Atur warna cluster
    unique_clusters = sorted(df['Cluster'].unique())
    cluster_colors = {cluster: COLOR_PALETTE[cluster % len(COLOR_PALETTE)] 
                    for cluster in unique_clusters}
    
    # Buat legend dalam bentuk 2 kolom, dengan keterangan rata kiri
    cols = st.columns(2)  
    for i, (cluster, color) in enumerate(cluster_colors.items()):
        kategori = df[df['Cluster'] == cluster]['Kategori'].iloc[0]
        with cols[i % 2]: 
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: flex-start; padding-left: 10px;">
                    <div style="width:20px; height:20px; background:{color}; 
                        border:1px solid black; margin-right:8px;"></div>
                    <div>{kategori}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    # Setup peta dasar
    m = folium.Map(location=[1, 20], zoom_start=1)

    for _, row in df.iterrows():
        # Informasi Popup
        volume_cols = [c for c in df.columns if 'Volume_' in c]
        nilai_cols = [c for c in df.columns if 'Nilai_' in c]
        
        avg_volume = row[volume_cols].mean() if volume_cols else 0
        avg_nilai = row[nilai_cols].mean() if nilai_cols else 0

        popup_html = f"""
        <b>{row['Negara Tujuan']}</b><br>
        Cluster: {row['Kategori']}<br>
        Rata-Rata:<br>
        • Volume: {avg_volume:,.0f} Ton<br>
        • Nilai: {avg_nilai:,.0f} USD<br>
        """

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=COLOR_PALETTE[row['Cluster'] % len(COLOR_PALETTE)]),
        ).add_to(m)
    
    folium_static(m)

def plot_evaluation_metrics(metrics, x_scaled):
    st.subheader("📊 Evaluasi Performa Clustering")
    
    cluster_range = range(2, 8)
    silhouette_scores = []
    dbi_scores = []
    processing_times = []

    for k in cluster_range:
        start_time = time.time()
        # Pilih model sesuai metode
        if metrics['method'] == 'KMeans':
            model = KMeans(n_clusters=k, max_iter=300, random_state=42, init='k-means++', n_init=10)
        elif metrics['method'] == 'KMedoids':
            model = KMedoids(n_clusters=k, max_iter=300, random_state=60, init='k-medoids++')
        else: 
            model = BisectingKMeans(n_clusters=k, max_iter=300, random_state=42)

        # Hitung waktu proses clustering
        # start_time = time.time()
        model.fit(x_scaled)
        processing_time = time.time() - start_time
        
        labels = model.labels_
        silhouette = silhouette_score(x_scaled, labels)
        dbi = davies_bouldin_score(x_scaled, labels)

        silhouette_scores.append(silhouette)
        dbi_scores.append(dbi)
        processing_times.append(processing_time)

    # Ambil waktu proses sesuai jumlah cluster yang dipilih
    selected_idx = list(cluster_range).index(metrics['n_clusters'])
    selected_processing_time = processing_times[selected_idx]

    # Tampilkan metrik dalam 3 kolom
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Metode", metrics['method'])
        st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}",
                help="Semakin mendekati 1 semakin baik")

    with col2:
        st.metric("Jumlah Cluster", metrics['n_clusters'])
        st.metric("Davies-Bouldin Index", f"{metrics['dbi']:.4f}",
                help="Semakin mendekati 0 semakin baik")

    with col3:
        st.metric("Waktu Proses", f"{selected_processing_time:.4f} detik")

    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # Silhouette plot
    sns.lineplot(x=list(cluster_range), y=silhouette_scores, 
                marker='o', ax=ax1, color='blue')
    ax1.axvline(x=metrics['n_clusters'], color='r', linestyle='--')
    ax1.set_title("Silhouette Score")
    ax1.set_xlabel("Jumlah Cluster")
    ax1.set_ylabel("Score")
    ax1.grid(True, linestyle='-', alpha=1)
    
    for i, k in enumerate(cluster_range):
        ax1.text(k, silhouette_scores[i], f"{silhouette_scores[i]:.4f}", 
                ha='center', va='bottom', fontsize=9)
    
    # DBI plot
    sns.lineplot(x=list(cluster_range), y=dbi_scores, 
                marker='o', ax=ax2, color='orange')
    ax2.axvline(x=metrics['n_clusters'], color='r', linestyle='--')
    ax2.set_title("Davies-Bouldin Index")
    ax2.set_xlabel("Jumlah Cluster")
    ax2.set_ylabel("Index")
    ax2.grid(True, linestyle='-', alpha=1)
    
    for i, k in enumerate(cluster_range):
        ax2.text(k, dbi_scores[i], f"{dbi_scores[i]:.4f}",
                ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_cluster_dist(df, selected_cluster=None):
    st.subheader("📊 Distribusi Fitur per Cluster")
    
    # ===== [1] Control Panel Section =====
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            year_pattern = r'_(\d{4})$'
            years = sorted(set(
                int(re.search(year_pattern, col).group(1))
                for col in df.columns
                if any(col.startswith(f) for f in FEATURE_EKSPOR) and re.search(year_pattern, col)
            ))
            year_range = st.slider(
                "Pilih range tahun:",
                min_value=min(years),
                max_value=max(years),
                value=(min(years), max(years))
            )
        with col2:
            cluster_options = ['Seluruh Cluster'] + df['Kategori'].unique().tolist()
            selected_cluster = st.selectbox(
                "Pilih Klaster:",
                options=cluster_options,
                index=0 if not selected_cluster else cluster_options.index(selected_cluster)
            )
        with col3:
            cols_per_row = st.slider(
                "Jumlah kolom per baris:",
                min_value=1,
                max_value=6,
                value=4
            )
    
    # ===== [2] Data Processing Section =====
    years_to_show = range(year_range[0], year_range[1] + 1)
    rows_needed = (len(years_to_show) + cols_per_row - 1) // cols_per_row
    
    # Normalize data
    df_norm = df.copy()
    scaler = StandardScaler()
    for year in years_to_show:
        for feature in FEATURE_EKSPOR:
            col = f"{feature}_{year}"
            if col in df_norm.columns:
                df_norm[col] = scaler.fit_transform(df_norm[[col]])
    
    # Filter data
    if selected_cluster != 'Seluruh Cluster':
        df_norm = df_norm[df_norm['Kategori'] == selected_cluster]
    
    # Prepare data for download
    all_melted = pd.DataFrame()
    for year in years_to_show:
        feature_cols = [f"{f}_{year}" for f in FEATURE_EKSPOR if f"{f}_{year}" in df_norm.columns]
        if feature_cols:
            melted = df_norm[['Kategori'] + feature_cols].melt(
                id_vars=['Kategori'],
                value_vars=feature_cols,
                var_name='Feature',
                value_name='Value'
            )
            melted['Tahun'] = year
            all_melted = pd.concat([all_melted, melted])
    
    # Calculate boxplot statistics
    if not all_melted.empty:
        boxplot_stats = all_melted.groupby(['Tahun', 'Kategori', 'Feature'])['Value'].describe()
        boxplot_stats = boxplot_stats.reset_index()
    
    # ===== [3] Visualization Section =====
    fig, axes = plt.subplots(
        rows_needed, 
        cols_per_row, 
        figsize=(6 * cols_per_row, 5 * rows_needed)
    )
    axes = axes.flatten() if rows_needed > 1 else [axes]
    palette = sns.color_palette("husl", len(FEATURE_EKSPOR))
    
    for i, year in enumerate(years_to_show):
        ax = axes[i]
        feature_cols = [f"{f}_{year}" for f in FEATURE_EKSPOR if f"{f}_{year}" in df_norm.columns]
        
        if not feature_cols:
            continue
            
        melted = df_norm[['Kategori'] + feature_cols].melt(
            id_vars=['Kategori'],
            value_vars=feature_cols,
            var_name='Feature',
            value_name='Value'
        )
        
        sns.boxplot(
            data=melted,
            x='Kategori',
            y='Value',
            hue='Feature',
            ax=ax,
            palette=palette
        )
        
        ax.set_title(f"Tahun {year}")
        ax.set_xlabel("")
        ax.set_ylabel("Nilai Standarisasi")
        ax.get_legend().remove()
    
    # Clean up empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    handles = [
        plt.Line2D([0], [0], color=palette[i], lw=4, label=feature)
        for i, feature in enumerate(FEATURE_EKSPOR)
    ]

    fig.legend(
        handles=handles,
        title='Fitur',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(FEATURE_EKSPOR),
        title_fontsize=15,
        fontsize=15
    )

    plt.tight_layout()
    
    # ===== [4] Download Section =====
    with st.container():
        
    # Buat 3 kolom: kiri, tengah (spasi), kanan
        col1, spacer, col2 = st.columns([2, 1, 2])

        with col1:
            # Tombol download gambar
            if fig:
                img_buffer = get_fig_buffer(fig)
                filename = f"distribusi_{selected_cluster.replace(' ', '_')}_{year_range[0]}-{year_range[1]}.png"
                st.download_button(
                    label="🖼️ Download PNG",
                    data=img_buffer,
                    file_name=filename,
                    mime="image/png",
                )

        with col2:
            # Tombol download manual book
            st.download_button(
                label="📕 Panduan membaca Box Plot",
                data=open('Manual Book/Panduan Box Plot.pdf', 'rb'),
                file_name="Panduan Box Plot.pdf",
                mime="application/pdf",
                help="Panduan membaca Box Plot"
            )

        # Tampilkan plot
        st.pyplot(fig)


    

def plot_ekspor_trends(df, default_metric="Volume", default_order="Terbesar ▶️", default_n_locations=5):
    st.subheader("📈 Tren Hasil Ekspor Panen Kakao")
    
    # Add search functionality
    with st.container():
        search_terms = st.text_input(
            "🔍 Bandingkan Negara Tujuan (pisahkan dengan koma):",
            placeholder="Contoh: Malaysia, India, Japan",
            help="Masukkan nama negara tujuan yang ingin dibandingkan trennya",
            key="trend_search"
        )
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric = st.selectbox(
                "Pilih Fitur:",
                options=FEATURE_EKSPOR,
                index=FEATURE_EKSPOR.index(default_metric),
                key="trend_metric"
            )
            
        with col2:
            # Disable sort order when searching
            sort_order = st.radio(
                "Urutan:",
                options=["Terbesar ▶️", "Terkecil ◀️"],
                index=0 if default_order == "Terbesar ▶️" else 1,
                horizontal=True,
                key="sort_order",
                disabled=bool(search_terms)  # Disabled when searching
            )
            
        with col3:
            max_locations = len(df)
            n_locations = st.slider(
                "Jumlah Negara:",
                min_value=1,
                max_value=min(20, max_locations),
                value=default_n_locations,
                key="num_locations",
                disabled=bool(search_terms)  # Disabled when searching
            )

    # Prepare data
    year_pattern = r'_(\d{4})$'
    metric_cols = [col for col in df.columns 
                if metric in col and re.search(year_pattern, col)]
    
    if not metric_cols:
        st.warning(f"⚠️ Data {metric} tidak tersedia")
        return
    
    years = sorted([int(re.search(year_pattern, col).group(1)) for col in metric_cols])
    
    # Filter data based on search terms if provided
    if search_terms:
        search_list = [term.strip() for term in search_terms.split(',') if term.strip()]
        
        # Check which countries exist and which don't
        found_countries = []
        missing_countries = []
        
        for country in search_list:
            matches = df[df['Negara Tujuan'].str.contains(country, case=False, na=False)]
            if not matches.empty:
                found_countries.extend(matches['Negara Tujuan'].unique().tolist())
            else:
                missing_countries.append(country)
        
        # Remove duplicates while preserving order
        seen = set()
        found_countries = [x for x in found_countries if not (x in seen or seen.add(x))]
        
        # Show warning for missing countries
        if missing_countries:
            st.warning(f"⚠️ Negara berikut tidak ditemukan: {', '.join(missing_countries)}")
        
        if not found_countries:
            st.error("❌ Tidak ada negara yang ditemukan dari pencarian Anda")
            return
            
        df_trend = df[df['Negara Tujuan'].isin(found_countries)]
        
        # In search mode, maintain the order of search terms
        df_trend['Match_Order'] = df_trend['Negara Tujuan'].apply(
            lambda x: next((i for i, country in enumerate(search_list) if country.lower() in x.lower()), len(search_list))
        )
        df_trend = df_trend.sort_values('Match_Order')
    else:
        df_trend = df.copy()
        df_trend['Average'] = df_trend[metric_cols].mean(axis=1)
        ascending = sort_order.startswith("Terkecil")
        df_trend = df_trend.sort_values('Average', ascending=ascending).head(n_locations)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5 + len(df_trend)*0.3))
    palette = sns.color_palette("tab20", len(df_trend))
    
    for i, (_, row) in enumerate(df_trend.iterrows()):
        ax.plot(
            years,
            row[metric_cols].values,
            marker='o',
            linestyle='-',
            color=palette[i % len(palette)],
            label=f"{row['Negara Tujuan']} ({row['Kategori']})",
            linewidth=2,
            markersize=6
        )
    
    # Style plot
    units = {"Volume": "Ton", "Nilai": "USD"}.get(metric, "")
    
    # Custom title based on mode
    if search_terms:
        display_countries = [country for country in search_list if any(country.lower() in found.lower() for found in found_countries)]
        title = f"TREN PERBANDINGAN: {', '.join(display_countries).upper()}\n"
        title += f"{metric} ({min(years)}-{max(years)})"
    else:
        title = f"{metric} Tren ({min(years)}-{max(years)})\n"
        title += f"{n_locations} {'Terbesar' if not ascending else 'Terkecil'} Negara"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Tahun")
    ax.set_ylabel(f"{metric} ({units})")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Improved legend placement
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8,
        title="Negara Tujuan",
        title_fontsize=9
    )
    
    plt.tight_layout()
    
    if fig:
        img_buffer = get_fig_buffer(fig)
        filename = f"Tren_{metric}_{min(years)}-{max(years)}.png"
        st.download_button(
            label="🖼️ Download PNG",
            data=img_buffer,
            file_name=filename,
            mime="image/png",
        )
    st.pyplot(fig)

def plot_top_ekspor(df, default_metric="Volume", default_order="Terbesar ▶️", default_n_locations=5):
    """Plot top export destination countries"""
    st.subheader("🏆 Top Negara Tujuan Hasil Panen Kakao")
    
    # Add search functionality
    with st.container():
        search_terms = st.text_input(
            "🔍 Bandingkan Negara (pisahkan dengan koma):",
            placeholder="Contoh: Malaysia, India, Japan",
            help="Masukkan nama negara yang ingin dibandingkan",
            key="search_input"
        )
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric = st.selectbox(
                "Pilih Fitur:",
                options=FEATURE_EKSPOR,
                index=FEATURE_EKSPOR.index(default_metric),
                key="top_metric"
            )
            
        with col2:
            # Sorting remains enabled during search
            sort_order = st.radio(
                "Urutan:",
                options=["Terbesar ▶️", "Terkecil ◀️"],
                index=0 if default_order == "Terbesar ▶️" else 1,
                horizontal=True,
                key="top_order"
            )
            
        with col3:
            # Only disable location count during search
            n_locations = st.slider(
                "Jumlah Negara:",
                min_value=1,
                max_value=min(20, len(df)),
                value=default_n_locations,
                key="top_num_locations",
                disabled=bool(search_terms)  # Disabled only during search
            )

    # Process data
    metric_cols = [col for col in df.columns if metric in col and re.search(r'_(\d{4})$', col)]
    
    if not metric_cols:
        st.warning(f"⚠️ Data untuk fitur '{metric}' tidak tersedia")
        return
    
    # Filter data based on search terms if provided
    if search_terms:
        search_list = [term.strip() for term in search_terms.split(',') if term.strip()]
        
        # Check which countries exist and which don't
        found_countries = []
        missing_countries = []
        
        for country in search_list:
            matches = df[df['Negara Tujuan'].str.contains(country, case=False, na=False)]
            if not matches.empty:
                found_countries.extend(matches['Negara Tujuan'].unique().tolist())
            else:
                missing_countries.append(country)
        
        # Remove duplicates while preserving order
        seen = set()
        found_countries = [x for x in found_countries if not (x in seen or seen.add(x))]
        
        # Show warning for missing countries
        if missing_countries:
            st.warning(f"⚠️ Negara berikut tidak ditemukan: {', '.join(missing_countries)}")
        
        if not found_countries:
            st.error("❌ Tidak ada negara yang ditemukan dari pencarian Anda")
            return
            
        df_filtered = df[df['Negara Tujuan'].isin(found_countries)]
    else:
        df_filtered = df.copy()
    
    df_top = df_filtered[['Negara Tujuan', 'Kategori'] + metric_cols].copy()
    df_top['Average'] = df_top[metric_cols].mean(axis=1)
    
    # Always apply sorting based on user selection
    ascending = sort_order.startswith("Terkecil")
    df_top = df_top.sort_values('Average', ascending=ascending)
    
    # Only limit number of locations when not in search mode
    if not search_terms:
        df_top = df_top.head(n_locations)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6 + len(df_top)*0.2))
    palette = sns.color_palette("tab20", len(df_top))
    location_colors = {loc: palette[i] for i, loc in enumerate(df_top['Negara Tujuan'])}
    
    bars = ax.bar(
        df_top['Negara Tujuan'],
        df_top['Average'],
        color=[location_colors[loc] for loc in df_top['Negara Tujuan']]
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    # Style plot
    units = {"Volume": "Ton", "Nilai": "USD"}[metric]
    years = [int(col.split('_')[-1]) for col in metric_cols]
    
    # Modify title based on mode
    if search_terms:
        display_countries = [c for c in search_list if any(c.lower() in f.lower() for f in found_countries)]
        title = f"PERBANDINGAN NEGARA: {', '.join(display_countries).upper()}\n"
        title += f"Berdasarkan Rata-Rata {metric.upper()} ({min(years)}-{max(years)})"
    else:
        title = f"TOP {n_locations} NEGARA TUJUAN BERDASARKAN RATA-RATA {metric.upper()}\n"
        title += f"Periode: {min(years)}-{max(years)}"
    
    ax.set_title(
        title,
        pad=20,
        fontsize=14,
        fontweight='bold'
    )
    
    ax.set_xlabel("Negara Tujuan", fontsize=12, labelpad=10)
    ax.set_ylabel(f"Rata-Rata ({units})", fontsize=12, labelpad=10)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # Create legend
    legend_handles = []
    for i, (_, row) in enumerate(df_top.iterrows()):
        legend_handles.append(
            plt.Line2D([0], [0], 
                    marker='o', 
                    color=palette[i % len(palette)],
                    label=f"{row['Negara Tujuan']} ({row['Kategori']})",
                    markersize=8,
                    linestyle='')
        )
    
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        title="Negara Tujuan",
        fontsize=9
    )
    plt.tight_layout()
    
    if fig:
        img_buffer = get_fig_buffer(fig)
        filename = f"Top_{metric}_{min(years)}-{max(years)}.png"
        st.download_button(
            label="🖼️ Download PNG",
            data=img_buffer,
            file_name=filename,
            mime="image/png",
        )
    st.pyplot(fig)

# ==============================================
# TAMPILAN UTAMA STREAMLIT
# ==============================================
st.title("📊 Clustering Lokasi Hasil Ekspor Panen Kakao dari Indonesia")
st.write("Unggah dataset dan lihat hasil clustering menggunakan KMeans, KMedoids, atau Bisecting KMeans beserta metrik evaluasinya.")

# Dataset information
st.subheader("🌍 Dataset Hasil Ekspor Panen Kakao")
st.markdown("""
    <div style='text-align: justify; text-indent: 40px;'>
            Dataset yang digunakan di halaman ini adalah data ekspor dari Indonesia ke negara tujuan. Data ini bersumber dari Badan Pusat Statistik yang mencatat aktivitas ekspor hasil perkebunan kakao secara tahunan. Fitur dataset tersebut meliputi Volume dan Nilai
            Volume (Ton) merupakan jumlah produksi kakao yang akan diekspor, sedangkan Nilai (USD) merupakan total harga ekspor kakao.
        </div>
""", unsafe_allow_html=True)

# Download buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="📥 Teamplate Dataset Ekspor Kakao",
        data=open('teamplate_dataset/Teamplate_Ekspor.xlsx', 'rb'),
        file_name='Teamplate_Ekspor.xlsx',
    )

with col2:
    st.download_button(
        label="📥 Dataset Ekspor ke Negara Tujuan",
        data=open('teamplate_dataset/Dataset_Ekspor.xlsx', 'rb'),
        file_name='Dataset_Ekspor.xlsx',
    )

with col3:
    st.download_button(
        label="📘 Buku Panduan Penggunaan Program",
        data=open('Manual Book/Manual Book Program Kakao.pdf', 'rb'),
        file_name='Manual_Book_Ekspor_Kakao.pdf',
    )

# File upload
uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)", 
    type=["xlsx"], 
    help="Pastikan format sesuai template yang disediakan"
)

# Sidebar settings
with st.sidebar:
    st.header("🔧 Pengaturan Clustering")
    method = st.radio(
        "Pilih Metode Clustering:", 
        ('KMeans', 'KMedoids', 'Bisecting KMeans')
    )
    n_clusters = st.slider(
        "Pilih Jumlah Cluster", 
        min_value=2, 
        max_value=7, 
        value=2
    )

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Validasi jumlah data (baris) minimal 10 lokasi
    if df.shape[0] < 10:
        st.error("❌ Jumlah data terlalu sedikit. Minimal harus ada 10 lokasi untuk proses clustering.")
    else:
        with st.spinner("Proses clustering..."):
            result = perform_clustering(uploaded_file, method, n_clusters)
            
            if result['error']:
                user_friendly_msg = translate_error_message(result['error']['message'])
                st.error("❌ Gagal memproses data: " + user_friendly_msg)
            else:
                st.success("✅ Clustering Berhasil!")
                df = result['dataframe']
                metrics = result['metrics']
                
                # kategori cluster
                df = categorize_clusters(df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📍 Hasil Klasterisasi")
                    st.dataframe(df[['Negara Tujuan', 'Cluster', 'Kategori']])
                
                # Visualisasi untuk jumlah cluster 
                cluster_counts = df.groupby(['Cluster', 'Kategori']).size().reset_index(name='Count')
                
                # Buat mapping warna berdasarkan kategori
                kategori_list = cluster_counts['Kategori'].unique()
                palette_dict = {
                    kategori: COLOR_PALETTE[i % len(COLOR_PALETTE)]
                    for i, kategori in enumerate(kategori_list)
                }

                with col2:
                    st.markdown("### 📍 Jumlah Data per Cluster")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        data=cluster_counts,
                        x='Kategori',
                        y='Count',
                        hue='Kategori',
                        palette=palette_dict,
                        ax=ax,
                        dodge=False
                    )
                    
                    for p in ax.patches:
                        ax.annotate(
                            f"{int(p.get_height())}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center',
                            va='center',
                            xytext=(0, 5),
                            textcoords='offset points'
                        )
                    
                    ax.set_title('Distribusi Cluster')
                    ax.set_xlabel('Kategori Cluster')
                    ax.set_ylabel('Jumlah')
                    ax.legend().set_visible(False)
                    st.pyplot(fig)
                
                # Fitur untuk mengunduh data hasil clustering
                st.subheader("💾 Download Hasil")
                categories = sorted(df['Kategori'].unique())
                selected_category = st.selectbox("Pilih Cluster:", ["Seluruh Cluster"] + categories)
                
                download_df = df if selected_category == "Seluruh Cluster" else df[df['Kategori'] == selected_category]
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    download_df.to_excel(writer, index=False)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.download_button(
                        label="📥 Download Excel",
                        data=output.getvalue(),
                        file_name=f"clustering_results_{selected_category.lower().replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Save gambar
                img_bytes = io.BytesIO()
                fig.savefig(img_bytes, format='png')
                img_bytes.seek(0)  

                with col4:
                    st.download_button(
                        label="📥 Download Visualisasi Cluster",
                        data=img_bytes,
                        file_name="cluster_visualization.png",
                        mime="image/png"
                    )
                    
                # Visualisasi
                display_cluster_map(df)
                plot_evaluation_metrics(metrics, result['x_scaled'])
                
                # analysis Lanjutan
                with st.expander("📈 Analisis Lanjutan", expanded=True):
                    tab1, tab2, tab3 = st.tabs(["Distribusi Cluster", "Tren Hasil Ekspor Panen", "Top Lokasi"])
                    
                    with tab1:
                            plot_cluster_dist(df)
                    
                    with tab2:
                            plot_ekspor_trends(df, default_metric="Volume")
                    
                    with tab3:
                            plot_top_ekspor(df, default_metric="Volume")

# Footer
show_footer()