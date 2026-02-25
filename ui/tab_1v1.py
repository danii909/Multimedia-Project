"""Tab 0: Confronto 1v1 – originale vs algoritmo scelto"""
import tempfile
import time
from pathlib import Path

import streamlit as st

from ui.stabilization import build_method_config
from ui.video_utils import combine_videos_grid, convert_to_web_compatible
from ui.plot_utils import create_trajectory_plot, create_displacement_histogram
from src.video_stabilizer import VideoStabilizer


# ──────────────────────────────────────────────────────────────
# Costanti
# ──────────────────────────────────────────────────────────────

_ALGO_OPTIONS = {
    "📦 Block Matching": "block_matching",
    "🌊 Optical Flow":   "optical_flow",
    "🎯 ORB Matching":   "orb_matching",
}

_TRANSFORM_LABELS = {
    "partial":     "Partial (traslazione)",
    "affine":      "Affine",
    "homography":  "Homography",
}


# ──────────────────────────────────────────────────────────────
# Helper session-state con prefisso isolato
# ──────────────────────────────────────────────────────────────

def _k(name):
    """Prefisso 1v1_ per isolare il session_state da altri tab."""
    return f"1v1_{name}"


def _get(key, default=None):
    return st.session_state.get(_k(key), default)


# ──────────────────────────────────────────────────────────────
# Costruzione metodo interno
# ──────────────────────────────────────────────────────────────

def _method_key(algo, transform):
    """Converte (algo, transform) nel metodo atteso da build_method_config."""
    if algo == "block_matching":
        return "block_matching"
    elif algo == "optical_flow":
        return f"optical_shi_tomasi_{transform}"
    else:
        return f"orb_matching_{transform}"


# ──────────────────────────────────────────────────────────────
# Step 1: Selezione algoritmo e trasformazione
# ──────────────────────────────────────────────────────────────

def _render_algo_and_transform():
    col1, col2 = st.columns(2)

    with col1:
        algo_label = st.selectbox(
            "🔬 Algoritmo di stabilizzazione",
            list(_ALGO_OPTIONS.keys()),
            key=_k("algo"),
        )

    algo = _ALGO_OPTIONS[algo_label]

    with col2:
        if algo == "block_matching":
            st.info("📦 Block Matching usa solo traslazione — nessuna trasformazione aggiuntiva disponibile.")
            transform = None
        else:
            transform = st.selectbox(
                "📐 Tipo di trasformazione globale",
                list(_TRANSFORM_LABELS.keys()),
                format_func=lambda x: _TRANSFORM_LABELS[x],
                key=_k("transform"),
            )

    return algo, transform


# ──────────────────────────────────────────────────────────────
# Step 2: Parametri specifici per algoritmo scelto
# ──────────────────────────────────────────────────────────────

def _render_params(algo, defaults, options):
    st.markdown("### ⚙️ Configurazione Parametri")

    with st.expander("🎨 Parametri Comuni (Smoothing & Compensation)", expanded=False):
        st.slider("Smoothing Window (frames)", 5, 60,
                  value=defaults['smoothing_window'], key=_k("smoothing_window"))
        st.selectbox("Tipo di Filtro", options['filter'],
                     index=_idx(options['filter'], defaults['filter_type']),
                     key=_k("filter_type"))
        st.slider("Crop Ratio", 0.75, 0.98,
                  value=defaults['crop_ratio'], step=0.01, key=_k("crop_ratio"))
        st.selectbox("Border Mode", options['border'],
                     index=_idx(options['border'], defaults['border_mode']),
                     key=_k("border_mode"))

    if algo == "block_matching":
        with st.expander("📦 Parametri Block Matching", expanded=False):
            st.select_slider("Block Size (px)", options=[8, 16, 32, 64],
                             value=defaults['block_size'], key=_k("block_size"))
            st.slider("Search Range (px)", 4, 32,
                      value=defaults['search_range'], step=2, key=_k("search_range"))
            st.selectbox("Metrica di Matching", options['metric'],
                         index=_idx(options['metric'], defaults['metric']),
                         key=_k("metric"))
            st.selectbox("Metodo di Aggregazione", options['aggregation'],
                         index=_idx(options['aggregation'], defaults['aggregation']),
                         key=_k("aggregation"))

    elif algo == "optical_flow":
        with st.expander("🌊 Parametri Optical Flow – Shi-Tomasi", expanded=False):
            st.slider("Max Corners", 200, 2000,
                      value=defaults['max_corners'], step=100, key=_k("max_corners"))
            st.slider("Quality Level", 0.001, 0.01,
                      value=defaults['quality_level'], step=0.001, format="%.3f",
                      key=_k("quality_level"))
            st.slider("Min Distance", 5, 20,
                      value=defaults['min_distance'], key=_k("min_distance"))
            st.select_slider("Window Size", options=[11, 15, 21, 25, 31],
                             value=defaults['win_size'], key=_k("win_size"))
            st.slider("RANSAC Threshold", 1.0, 15.0,
                      value=defaults['ransac_threshold_st'], step=0.5,
                      key=_k("ransac_threshold_st"))

    else:  # orb_matching
        with st.expander("🎯 Parametri ORB Matching", expanded=False):
            st.slider("N° Features", 200, 1000,
                      value=defaults['orb_n_features'], step=50, key=_k("orb_n_features"))
            st.slider("Scale Factor", 1.1, 1.5,
                      value=defaults['orb_scale_factor'], step=0.05,
                      key=_k("orb_scale_factor"))
            st.slider("Pyramid Levels", 4, 12,
                      value=defaults['orb_n_levels'], key=_k("orb_n_levels"))
            st.slider("Ratio Threshold", 0.5, 0.9,
                      value=defaults['orb_ratio_threshold'], step=0.05,
                      key=_k("orb_ratio_threshold"))
            st.slider("RANSAC Threshold", 1.0, 15.0,
                      value=defaults['ransac_threshold_orb'], step=0.5,
                      key=_k("ransac_threshold_orb"))


# ──────────────────────────────────────────────────────────────
# Lettura parametri da session_state
# ──────────────────────────────────────────────────────────────

def _read_params(defaults, transform):
    def g(k, dk=None):
        return st.session_state.get(_k(k), defaults[dk or k])

    return {
        'smoothing_window':     g('smoothing_window'),
        'filter_type':          g('filter_type'),
        'crop_ratio':           g('crop_ratio'),
        'border_mode':          g('border_mode'),
        'block_size':           g('block_size'),
        'search_range':         g('search_range'),
        'metric':               g('metric'),
        'aggregation':          g('aggregation'),
        'estimate_rotation_bm': transform in ('affine', 'homography'),
        'max_corners':          g('max_corners'),
        'quality_level':        g('quality_level'),
        'min_distance':         g('min_distance'),
        'win_size':             g('win_size'),
        'ransac_threshold_st':  g('ransac_threshold_st'),
        'orb_n_features':       g('orb_n_features'),
        'orb_scale_factor':     g('orb_scale_factor'),
        'orb_n_levels':         g('orb_n_levels'),
        'orb_ratio_threshold':  g('orb_ratio_threshold'),
        'ransac_threshold_orb': g('ransac_threshold_orb'),
    }


# ──────────────────────────────────────────────────────────────
# Esecuzione stabilizzazione 1v1
# ──────────────────────────────────────────────────────────────

def _run_1v1(method_key, input_path, params):
    progress_bar = st.progress(0.0)
    status_text  = st.empty()

    def _cb(frac):
        progress_bar.progress(min(frac, 1.0))
        status_text.text(f"⏳ Elaborazione: {int(frac * 100)}%")

    config     = build_method_config(method_key, params)
    start_time = time.time()
    stabilizer = VideoStabilizer(config)

    with tempfile.NamedTemporaryFile(delete=False, suffix='_1v1.mp4') as tmp:
        output_path = tmp.name

    success = stabilizer.stabilize_video(input_path, output_path, progress_callback=_cb)
    elapsed = time.time() - start_time

    progress_bar.progress(1.0)
    status_text.empty()

    if success:
        web_output = output_path.replace('.mp4', '_web.mp4')
        ok          = convert_to_web_compatible(output_path, web_output)
        return {
            'output_path':     output_path,
            'display_path':    web_output if ok else output_path,
            'metrics':         stabilizer.get_metrics(),
            'processing_time': elapsed,
            'success':         True,
        }
    else:
        return {'success': False, 'error': 'Stabilization failed'}


# ──────────────────────────────────────────────────────────────
# Sezione video comparativo
# ──────────────────────────────────────────────────────────────

def _render_video_comparison(input_path, result):
    st.markdown("### 🎬 Confronto Video (Originale vs Stabilizzato)")

    with st.spinner("⏳ Generazione video comparativo..."):
        combined_path = combine_videos_grid(
            [input_path, result['display_path']],
            ['Originale', 'Stabilizzato'],
        )

    if combined_path and Path(combined_path).exists():
        web_combined = combined_path.replace('.mp4', '_web.mp4')
        ok           = convert_to_web_compatible(combined_path, web_combined)
        display      = web_combined if ok else combined_path
        st.video(display)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            with open(combined_path, 'rb') as f:
                st.download_button(
                    "⬇️ Scarica Video Comparativo",
                    data=f,
                    file_name="1v1_comparison.mp4",
                    mime="video/mp4",
                    key="1v1_download_video",
                )
        with col_dl2:
            out_path = result.get('output_path', result['display_path'])
            with open(out_path, 'rb') as f:
                st.download_button(
                    "⬇️ Scarica Video Stabilizzato",
                    data=f,
                    file_name="video_stabilizzato.mp4",
                    mime="video/mp4",
                    key="1v1_download_stabilized",
                )
    else:
        # Fallback: due player affiancati
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📹 Originale**")
            st.video(input_path)
        with col2:
            st.markdown("**✅ Stabilizzato**")
            st.video(result['display_path'])
            out_path = result.get('output_path', result['display_path'])
            with open(out_path, 'rb') as f:
                st.download_button(
                    "⬇️ Scarica Video Stabilizzato",
                    data=f,
                    file_name="stabilizzato.mp4",
                    mime="video/mp4",
                    key="1v1_download_stabilized_fallback",
                )


# ──────────────────────────────────────────────────────────────
# Sezione metriche
# ──────────────────────────────────────────────────────────────

def _render_metrics_1v1(metrics, processing_time):
    st.markdown("### 📊 Risultati")

    # Avviso se la smoothing window è stata clampata
    if metrics.get('smoothing_window_clamped'):
        sw_req  = metrics.get('smoothing_window_requested', '?')
        sw_used = metrics.get('smoothing_window_used', '?')
        st.warning(
            f"⚠️ Smoothing window ridotta automaticamente da **{sw_req}** a **{sw_used}** frame "
            f"(il video ha {metrics.get('num_frames', '?')} frame). "
            f"I valori di jitter reduction sono calcolati sulla window effettiva."
        )

    # Jitter reduction
    st.markdown("#### 📉 Riduzione del Jitter")
    c1, c2, c3 = st.columns(3)
    for col, key, label in [
        (c1, 'jitter_reduction_x',     'Jitter Reduction X'),
        (c2, 'jitter_reduction_y',     'Jitter Reduction Y'),
        (c3, 'jitter_reduction_angle', 'Jitter Reduction Angle'),
    ]:
        val = metrics.get(key, 0)
        col.metric(
            label,
            f"{val:.1f}%",
            delta=f"{val:.1f}%" if val > 0 else None,
            delta_color="normal" if val > 0 else "inverse",
        )

    st.markdown("---")

    # RMS e tempo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMS X",            f"{metrics.get('rms_dx',    0):.2f} px")
    c2.metric("RMS Y",            f"{metrics.get('rms_dy',    0):.2f} px")
    c3.metric("RMS Rotation",     f"{metrics.get('rms_angle', 0):.3f}°")
    c4.metric("Processing Time",  f"{processing_time:.2f}s")

    st.markdown("---")

    # Traiettoria
    st.markdown("#### 📈 Traiettoria Raw vs Smoothed")
    fig_traj = create_trajectory_plot(metrics)
    if fig_traj:
        st.pyplot(fig_traj)
    else:
        st.info("Dati traiettoria non disponibili")

    st.markdown("---")

    '''
    # Istogrammi spostamenti
    st.markdown("#### 📊 Istogrammi degli Spostamenti Frame-by-Frame")
    fig_hist = create_displacement_histogram(metrics)
    if fig_hist:
        st.pyplot(fig_hist)
    else:
        st.info("Dati per istogramma non disponibili")

    st.markdown("---")
    '''
    with st.expander("🔍 Dettagli Tecnici Completi"):
        st.json(metrics)


# ──────────────────────────────────────────────────────────────
# Entry point del tab
# ──────────────────────────────────────────────────────────────

def render_tab_1v1(configs):
    """Renderizza Tab 0 – Confronto 1v1."""
    if 'input_path' not in st.session_state:
        st.info("📹 Carica prima un video dalla sidebar")
        return

    d = configs['defaults']
    o = configs['options']

    st.markdown("### 🔬 Scelta Algoritmo e Trasformazione")
    algo, transform = _render_algo_and_transform()

    st.markdown("---")
    _render_params(algo, d, o)

    st.markdown("---")
    _, btn_col, _ = st.columns([2, 2, 2])
    with btn_col:
        avvia = st.button(
            "▶️ Avvia Stabilizzazione",
            type="primary",
            key="1v1_run_button",
            width='stretch',
        )

    if avvia:
        method_key = _method_key(algo, transform)
        params     = _read_params(d, transform)
        result     = _run_1v1(method_key, st.session_state['input_path'], params)
        if result['success']:
            st.session_state['1v1_result'] = result
            st.success(f"✅ Completato in {result['processing_time']:.2f}s!")
        else:
            st.error(f"❌ Errore: {result.get('error', 'Sconosciuto')}")
            st.session_state.pop('1v1_result', None)

    if st.session_state.get('1v1_result', {}).get('success'):
        result = st.session_state['1v1_result']
        st.markdown("---")
        _render_video_comparison(st.session_state['input_path'], result)
        st.markdown("---")
        _render_metrics_1v1(result['metrics'], result['processing_time'])


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _idx(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return 0
