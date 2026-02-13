"""
Streamlit App per Video Stabilization System
App interattiva per stabilizzare video con configurazione dinamica dei parametri
"""

import streamlit as st
import tempfile
import json
import yaml
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import time
from src.video_stabilizer import VideoStabilizer


# Configurazione pagina
st.set_page_config(
    page_title="Video Stabilization App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato per migliorare l'aspetto
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    
    /* Ridimensiona i video player */
    video {
        max-height: 350px !important;
        width: 100% !important;
        object-fit: contain;
    }
    
    /* Allinea le colonne verticalmente */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        align-items: stretch;
    }
    </style>
""", unsafe_allow_html=True)


def create_trajectory_plot(metrics):
    """Crea grafico della traiettoria raw vs smoothed"""
    if not metrics or 'raw_trajectory' not in metrics:
        return None
    
    raw = np.array(metrics['raw_trajectory'])
    smooth = np.array(metrics['smoothed_trajectory'])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # X displacement
    axes[0].plot(raw[:, 0], label='Raw X', alpha=0.7, linewidth=1)
    axes[0].plot(smooth[:, 0], label='Smoothed X', linewidth=2)
    axes[0].set_ylabel('X Displacement (px)')
    axes[0].set_title('Traiettoria X')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y displacement
    axes[1].plot(raw[:, 1], label='Raw Y', alpha=0.7, linewidth=1)
    axes[1].plot(smooth[:, 1], label='Smoothed Y', linewidth=2)
    axes[1].set_ylabel('Y Displacement (px)')
    axes[1].set_xlabel('Frame')
    axes[1].set_title('Traiettoria Y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_metrics_comparison(metrics):
    """Crea grafico a barre delle metriche principali"""
    if not metrics:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    metric_names = ['RMS X', 'RMS Y', 'RMS Angle', 'Jitter X', 'Jitter Y', 'Jitter Angle']
    values = [
        metrics.get('rms_dx', 0),
        metrics.get('rms_dy', 0),
        metrics.get('rms_angle', 0) * 10,  # Scala per visualizzazione
        metrics.get('jitter_reduction_x', 0),
        metrics.get('jitter_reduction_y', 0),
        metrics.get('jitter_reduction_angle', 0)
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(metric_names, values, color=colors, alpha=0.7)
    
    ax.set_ylabel('Valore')
    ax.set_title('Metriche di Stabilizzazione')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Aggiungi valori sopra le barre
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def get_video_info(video_path):
    """Estrae informazioni base dal video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }


def convert_to_web_compatible(input_path, output_path):
    """Converte il video in formato compatibile con i browser web (H.264)"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prova diversi codec H.264 fino a trovarne uno che funziona
        codecs_to_try = ['avc1', 'h264', 'x264', 'H264']
        out = None
        
        for codec_str in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
                out.release()
                out = None
            except:
                continue
        
        # Se nessun codec H.264 funziona, usa mp4v
        if out is None or not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return False
        
        # Copia frame per frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        return True
        
    except Exception as e:
        st.warning(f"Conversione codec non riuscita: {e}")
        return False


def main():
    # Carica configurazioni dai file YAML
    config_dir = Path(__file__).parent / 'config'
    
    with open(config_dir / 'config_block_matching.yaml', 'r') as f:
        config_block = yaml.safe_load(f)
    
    with open(config_dir / 'config_optical_flow.yaml', 'r') as f:
        config_optical = yaml.safe_load(f)
    
    # Header
    st.markdown('<p class="main-header">🎬 Video Stabilization System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Stabilizza i tuoi video con algoritmi avanzati di motion estimation</p>', unsafe_allow_html=True)
    
    # Sidebar - Configurazione
    st.sidebar.title("⚙️ Configurazione")
    
    # Selezione metodo
    st.sidebar.subheader("🔬 Metodo di Stima del Movimento")
    estimation_method = st.sidebar.radio(
        "Seleziona metodo:",
        ["Block Matching", "Optical Flow"],
        help="Block Matching: più veloce, robusto su scene uniformi\nOptical Flow: più preciso su feature distintive"
    )
    
    method_key = 'block_matching' if estimation_method == "Block Matching" else 'optical_flow'
    
    st.sidebar.markdown("---")
    
    # Parametri specifici per metodo
    if estimation_method == "Block Matching":
        st.sidebar.subheader("📦 Parametri Block Matching")
        
        # Valori di default dal config
        default_block_size = config_block['motion_estimation']['block_size']
        default_search_range = config_block['motion_estimation']['search_range']
        default_metric = config_block['motion_estimation']['metric']
        default_aggregation = config_block['global_motion']['aggregation_method']
        
        block_size = st.sidebar.select_slider(
            "Block Size (px)",
            options=[8, 16, 32, 64],
            value=default_block_size,
            help="Dimensione dei blocchi per il matching. Valori più alti = più robusto ma meno dettagliato"
        )
        
        search_range = st.sidebar.slider(
            "Search Range (px)",
            min_value=4,
            max_value=32,
            value=default_search_range,
            step=2,
            help="Range di ricerca per il matching. Valori più alti = cattura movimenti più grandi"
        )
        
        metric_options = ["sad", "mad", "ssd", "ncc"]
        metric = st.sidebar.selectbox(
            "Metrica di Matching",
            metric_options,
            index=metric_options.index(default_metric) if default_metric in metric_options else 0,
            help="SAD: Sum of Absolute Differences (veloce)\nMAD: Mean Absolute Difference\nSSD: Sum of Squared Differences\nNCC: Normalized Cross Correlation (più robusto)"
        )
        
        aggregation_options = ["median", "mean", "weighted"]
        aggregation = st.sidebar.selectbox(
            "Metodo di Aggregazione",
            aggregation_options,
            index=aggregation_options.index(default_aggregation) if default_aggregation in aggregation_options else 0,
            help="Metodo per aggregare i motion vectors locali"
        )
        
    else:  # Optical Flow
        st.sidebar.subheader("🌊 Parametri Optical Flow")
        
        # Valori di default dal config
        default_max_corners = config_optical['global_motion']['optical_flow']['max_corners']
        default_quality_level = config_optical['global_motion']['optical_flow']['quality_level']
        default_min_distance = config_optical['global_motion']['optical_flow']['min_distance']
        default_win_size = config_optical['global_motion']['optical_flow']['win_size']
        default_ransac_threshold = config_optical['global_motion']['optical_flow']['ransac_reproj_threshold']
        
        max_corners = st.sidebar.slider(
            "Max Corners",
            min_value=200,
            max_value=2000,
            value=default_max_corners,
            step=100,
            help="Numero massimo di feature points da rilevare"
        )
        
        quality_level = st.sidebar.slider(
            "Quality Level",
            min_value=0.001,
            max_value=0.01,
            value=default_quality_level,
            step=0.001,
            format="%.3f",
            help="Soglia di qualità per l'accettazione dei corner"
        )
        
        min_distance = st.sidebar.slider(
            "Min Distance",
            min_value=5,
            max_value=20,
            value=default_min_distance,
            help="Distanza minima tra feature points (px)"
        )
        
        win_size = st.sidebar.select_slider(
            "Window Size",
            options=[11, 15, 21, 25, 31],
            value=default_win_size,
            help="Dimensione finestra per Lucas-Kanade tracking"
        )
        
        ransac_threshold = st.sidebar.slider(
            "RANSAC Threshold",
            min_value=1.0,
            max_value=15.0,
            value=default_ransac_threshold,
            step=0.5,
            help="Soglia per rimozione outlier con RANSAC"
        )
    
    st.sidebar.markdown("---")
    
    # Parametri comuni
    st.sidebar.subheader("🎨 Trajectory Smoothing")
    
    # Valori comuni da config (uguali in entrambi i file)
    default_smoothing_window = config_block['trajectory_smoothing']['smoothing_window']
    default_filter_type = config_block['trajectory_smoothing']['filter_type']
    
    smoothing_window = st.sidebar.slider(
        "Smoothing Window (frames)",
        min_value=5,
        max_value=60,
        value=default_smoothing_window,
        help="Finestra temporale per lo smoothing della traiettoria"
    )
    
    filter_options = ["moving_average", "gaussian", "exponential"]
    filter_type = st.sidebar.selectbox(
        "Tipo di Filtro",
        filter_options,
        index=filter_options.index(default_filter_type) if default_filter_type in filter_options else 0,
        help="Filtro per lo smoothing temporale"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("🖼️ Motion Compensation")
    
    # Valori di default dal config
    default_crop_ratio = config_block['motion_compensation']['crop_ratio']
    default_border_mode = config_block['motion_compensation']['border_mode']
    default_estimate_rotation = config_block['global_motion']['estimate_rotation']
    
    crop_ratio = st.sidebar.slider(
        "Crop Ratio",
        min_value=0.75,
        max_value=0.98,
        value=default_crop_ratio,
        step=0.01,
        help="Rapporto di crop per eliminare bordi neri (più alto = meno crop)"
    )
    
    border_options = ["constant", "replicate", "reflect"]
    border_mode = st.sidebar.selectbox(
        "Border Mode",
        border_options,
        index=border_options.index(default_border_mode) if default_border_mode in border_options else 0,
        help="Modalità di gestione dei bordi"
    )
    
    estimate_rotation = st.sidebar.checkbox(
        "Stima Rotazione",
        value=default_estimate_rotation,
        help="Abilita stima e compensazione della rotazione"
    )
    
    # Area principale
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Metriche & Analisi", "ℹ️ Info"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📹 Video Input")
            uploaded_file = st.file_uploader(
                "Carica un video instabile",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Supportati: MP4, AVI, MOV, MKV"
            )
            
            if uploaded_file is not None:
                # Salva file temporaneo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                    tmp_input.write(uploaded_file.read())
                    input_path = tmp_input.name
                
                # Mostra info video
                video_info = get_video_info(input_path)
                if video_info:
                    st.success("✅ Video caricato con successo!")
                    
                    info_col1, info_col2, info_col3 = st.columns(3)
                    info_col1.metric("📐 Risoluzione", f"{video_info['width']}x{video_info['height']}")
                    info_col2.metric("🎞️ Frame", video_info['frame_count'])
                    info_col3.metric("⏱️ Durata", f"{video_info['duration']:.1f}s")
                    
                    # Store in session state
                    st.session_state['input_path'] = input_path
                    st.session_state['video_info'] = video_info
                else:
                    st.error("❌ Errore nel caricamento del video")
        
        with col2:
            st.subheader("🎬 Stabilizzazione")
            
            if uploaded_file is not None and 'input_path' in st.session_state:
                # Bottone di stabilizzazione
                if st.button("▶️ Avvia Stabilizzazione", type="primary"):
                    # Crea configurazione
                    config = {
                        'global_motion': {
                            'estimation_method': method_key,
                            'motion_model': 'affine' if estimate_rotation else 'translation',
                            'aggregation_method': aggregation if estimation_method == "Block Matching" else 'median',
                            'outlier_threshold': 2.0,
                            'estimate_rotation': estimate_rotation
                        },
                        'trajectory_smoothing': {
                            'smoothing_window': smoothing_window,
                            'filter_type': filter_type,
                            'deadband_px': 0.0
                        },
                        'motion_compensation': {
                            'crop_ratio': crop_ratio,
                            'border_mode': border_mode
                        }
                    }
                    
                    if estimation_method == "Block Matching":
                        config['motion_estimation'] = {
                            'block_size': block_size,
                            'search_range': search_range,
                            'metric': metric
                        }
                    else:
                        config['global_motion']['optical_flow'] = {
                            'max_corners': max_corners,
                            'quality_level': quality_level,
                            'min_distance': min_distance,
                            'win_size': win_size,
                            'ransac_reproj_threshold': ransac_threshold,
                            'ransac_confidence': 0.995
                        }
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Stabilizzazione
                        status_text.text("🔄 Inizializzazione...")
                        stabilizer = VideoStabilizer(config)
                        
                        # Output temporaneo
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                            output_path = tmp_output.name
                        
                        status_text.text("🎬 Stabilizzazione in corso...")
                        start_time = time.time()
                        
                        success = stabilizer.stabilize_video(
                            st.session_state['input_path'],
                            output_path
                        )
                        
                        elapsed_time = time.time() - start_time
                        
                        if success:
                            progress_bar.progress(100)
                            status_text.text("✅ Stabilizzazione completata!")
                            
                            # Converti in formato web-compatible
                            status_text.text("🔄 Conversione per visualizzazione web...")
                            web_output_path = output_path.replace('.mp4', '_web.mp4')
                            
                            conversion_success = convert_to_web_compatible(output_path, web_output_path)
                            
                            if conversion_success:
                                # Usa il video convertito per visualizzazione
                                display_path = web_output_path
                            else:
                                # Fallback al video originale
                                display_path = output_path
                            
                            # Salva metriche e output in session state
                            st.session_state['output_path'] = output_path  # Per download
                            st.session_state['display_path'] = display_path  # Per visualizzazione
                            st.session_state['metrics'] = stabilizer.get_metrics()
                            st.session_state['processing_time'] = elapsed_time
                            
                            status_text.text("✅ Completato!")
                            st.success(f"🎉 Video stabilizzato in {elapsed_time:.2f} secondi!")
                            
                            # Download button
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Scarica Video Stabilizzato",
                                    data=f,
                                    file_name=f"stabilized_{uploaded_file.name}",
                                    mime="video/mp4"
                                )
                        else:
                            progress_bar.progress(0)
                            status_text.text("❌ Errore durante la stabilizzazione")
                            st.error("Si è verificato un errore durante la stabilizzazione")
                    
                    except Exception as e:
                        progress_bar.progress(0)
                        status_text.text("❌ Errore")
                        st.error(f"Errore: {str(e)}")
            else:
                st.info("👆 Carica un video per iniziare")
        
        # Sezione video allineata verticalmente - spazio dedicato uguale per entrambi
        if uploaded_file is not None:
            st.markdown("---")
            st.markdown("### 📺 Confronto Video")
            
            video_col1, video_col2 = st.columns([1, 1])
            
            with video_col1:
                st.markdown("**Video Originale**")
                # Container per mantenere l'altezza uniforme
                video_container1 = st.container()
                with video_container1:
                    st.video(uploaded_file)
            
            with video_col2:
                st.markdown("**Video Stabilizzato**")
                # Container per mantenere l'altezza uniforme
                video_container2 = st.container()
                with video_container2:
                    if 'output_path' in st.session_state:
                        video_path = st.session_state.get('display_path', st.session_state['output_path'])
                        st.video(video_path)
                    else:
                        # Placeholder con altezza equivalente
                        st.info("👈 Premi 'Avvia Stabilizzazione' per vedere il risultato")
    with tab2:
        st.subheader("📊 Analisi delle Metriche")
        
        if 'metrics' in st.session_state:
            metrics = st.session_state['metrics']
            
            # Metriche principali
            st.markdown("### 📈 Metriche di Stabilità")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "RMS Displacement X",
                    f"{metrics.get('rms_dx', 0):.2f} px",
                    help="Root Mean Square dello spostamento orizzontale"
                )
            
            with col2:
                st.metric(
                    "RMS Displacement Y",
                    f"{metrics.get('rms_dy', 0):.2f} px",
                    help="Root Mean Square dello spostamento verticale"
                )
            
            with col3:
                st.metric(
                    "RMS Rotation",
                    f"{metrics.get('rms_angle', 0):.3f}°",
                    help="Root Mean Square della rotazione"
                )
            
            with col4:
                st.metric(
                    "Processing Time",
                    f"{st.session_state.get('processing_time', 0):.2f}s",
                    help="Tempo totale di elaborazione"
                )
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                jitter_x = metrics.get('jitter_reduction_x', 0)
                delta_color = "normal" if jitter_x > 0 else "inverse"
                st.metric(
                    "Jitter Reduction X",
                    f"{jitter_x:.1f}%",
                    delta=f"{jitter_x:.1f}%" if jitter_x > 0 else None,
                    delta_color=delta_color,
                    help="Riduzione della varianza in X (raw → smoothed)"
                )
            
            with col2:
                jitter_y = metrics.get('jitter_reduction_y', 0)
                delta_color = "normal" if jitter_y > 0 else "inverse"
                st.metric(
                    "Jitter Reduction Y",
                    f"{jitter_y:.1f}%",
                    delta=f"{jitter_y:.1f}%" if jitter_y > 0 else None,
                    delta_color=delta_color,
                    help="Riduzione della varianza in Y (raw → smoothed)"
                )
            
            with col3:
                jitter_angle = metrics.get('jitter_reduction_angle', 0)
                delta_color = "normal" if jitter_angle > 0 else "inverse"
                st.metric(
                    "Jitter Reduction Angle",
                    f"{jitter_angle:.1f}%",
                    delta=f"{jitter_angle:.1f}%" if jitter_angle > 0 else None,
                    delta_color=delta_color,
                    help="Riduzione della varianza nella rotazione"
                )
            
            st.markdown("---")
            
            # Grafici
            st.markdown("### 📉 Visualizzazioni")
            
            st.markdown("#### Traiettoria Raw vs Smoothed")
            fig_traj = create_trajectory_plot(metrics)
            if fig_traj:
                st.pyplot(fig_traj)
            else:
                st.info("Dati traiettoria non disponibili")
            
            st.markdown("---")
            
            # Dettagli tecnici espandibili
            with st.expander("🔍 Dettagli Tecnici Completi"):
                st.json(metrics)
            
            # Export metriche
            metrics_json = json.dumps(metrics, indent=2)
            st.download_button(
                label="📥 Scarica Metriche (JSON)",
                data=metrics_json,
                file_name="stabilization_metrics.json",
                mime="application/json"
            )
        
        else:
            st.info("📊 Esegui prima la stabilizzazione per vedere le metriche")
    
    with tab3:
        st.subheader("ℹ️ Informazioni sul Sistema")
        
        st.markdown("""
        ### 🎬 Video Stabilization System
        
        Sistema avanzato di stabilizzazione video che implementa due approcci distinti:
        
        #### 🔷 Block Matching
        - **Principio**: Divide il frame in blocchi e cerca il miglior match nel frame successivo
        - **Vantaggi**: Veloce, robusto su scene uniformi, non richiede feature detection
        - **Ideale per**: Video con poca texture, scene uniformi, processing rapido
        
        #### 🔶 Optical Flow (Lucas-Kanade)
        - **Principio**: Rileva feature distintive e le traccia tra frame consecutivi
        - **Vantaggi**: Alta precisione, gestione robusta outlier (RANSAC), stima rotazione
        - **Ideale per**: Scene con molti dettagli, movimenti complessi, massima qualità
        
        ---
        
        ### 📊 Pipeline di Stabilizzazione
        
        1. **Motion Estimation**: Calcola il movimento tra frame consecutivi
        2. **Global Motion Estimation**: Aggrega i motion vectors per stimare il movimento della camera
        3. **Trajectory Smoothing**: Filtra temporalmente la traiettoria per eliminare il jitter
        4. **Motion Compensation**: Applica trasformazioni inverse per stabilizzare
        
        ---
        
        ### 🎯 Consigli per l'Uso
        
        **Parametri Block Matching**:
        - `Block Size` più grande → più robusto ma meno dettagliato
        - `Search Range` più grande → cattura movimenti più ampi
        - `SAD` → veloce, `NCC` → più robusto a illuminazione
        
        **Parametri Optical Flow**:
        - `Max Corners` più alto → più feature, maggiore precisione
        - `Quality Level` più basso → più corner accettati
        - `RANSAC Threshold` più basso → più rigido nel filtrare outlier
        
        **Smoothing**:
        - `Smoothing Window` più grande → più smooth ma rischio lag
        - `Moving Average` → semplice e veloce
        - `Gaussian` → smooth più naturale
        
        **Compensation**:
        - `Crop Ratio` più alto → meno crop ma possibili bordi neri
        - `Crop Ratio` più basso → più crop ma nessun bordo nero
        
        ---
        
        ### 📚 Riferimenti
        
        - **Block Matching**: Jain & Jain, IEEE Trans. 1981
        - **Lucas-Kanade**: Lucas & Kanade, IJCAI 1981
        - **RANSAC**: Fischler & Bolles, Comm. ACM 1981
        
        ---
        
        ### 👨‍💻 Sviluppatori
        
        Progetto Multimedia - Video Stabilization
        
        [GitHub Repository](https://github.com/danii909/Multimedia-Project)
        """)
        
        st.markdown("---")
        
        # System info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 Versione**: 1.0.0")
            st.markdown("**🐍 Python**: 3.8+")
            st.markdown("**📊 Framework**: Streamlit")
        
        with col2:
            st.markdown("**📦 OpenCV**: 4.5+")
            st.markdown("**🔢 NumPy**: 1.21+")
            st.markdown("**📄 Licenza**: MIT")


if __name__ == "__main__":
    main()
