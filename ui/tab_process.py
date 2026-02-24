"""Tab 1: Upload & Process – selezione metodi, configurazione parametri, avvio e risultati"""
import streamlit as st
from pathlib import Path
from ui.stabilization import METHOD_NAMES, run_all_methods, read_params_from_session
from ui.video_utils import combine_videos_grid, convert_to_web_compatible


# ──────────────────────────────────────────────────────────────
# Selezione metodi
# ──────────────────────────────────────────────────────────────

def render_method_selection():
    """Mostra i checkbox per scegliere i metodi. Restituisce la lista dei metodi selezionati."""
    st.markdown("### 🎯 Selezione dei metodi")

    col_lk, col_orb, col_bm = st.columns(3)

    with col_lk:
        st.markdown("##### Optical Flow – Corner Tracking")
        use_lk_partial    = st.checkbox("🌊 Optical Flow (LK) - Partial",     value=False)
        use_lk_affine     = st.checkbox("🌊 Optical Flow (LK) - Affine",      value=False)
        use_lk_homography = st.checkbox("🌊 Optical Flow (LK) - Homography",  value=False)

    with col_orb:
        st.markdown("##### Feature Matching – ORB")
        use_orb_partial    = st.checkbox("🎯 ORB Matching - Partial",    value=False)
        use_orb_affine     = st.checkbox("🎯 ORB Matching - Affine",     value=False)
        use_orb_homography = st.checkbox("🎯 ORB Matching - Homography", value=False)

    with col_bm:
        st.markdown("##### Block-Based Motion Estimation")
        use_block_matching = st.checkbox("📦 Block Matching", value=False)

    selected = []
    if use_block_matching:     selected.append("block_matching")
    if use_lk_partial:         selected.append("optical_shi_tomasi_partial")
    if use_lk_affine:          selected.append("optical_shi_tomasi_affine")
    if use_lk_homography:      selected.append("optical_shi_tomasi_homography")
    if use_orb_partial:        selected.append("optical_orb_partial")
    if use_orb_affine:         selected.append("optical_orb_affine")
    if use_orb_homography:     selected.append("optical_orb_homography")

    return selected


# ──────────────────────────────────────────────────────────────
# Configurazione parametri
# ──────────────────────────────────────────────────────────────

def render_params_section(selected_methods, configs):
    """Mostra i pannelli di configurazione parametri (expander)."""
    d = configs['defaults']
    o = configs['options']

    st.markdown("### ⚙️ Configurazione Parametri")

    with st.expander("🎨 Parametri Comuni (Smoothing & Compensation)", expanded=False):
        st.slider("Smoothing Window (frames)", 5, 60,
                  value=d['smoothing_window'], key="comp_smoothing_window")
        st.selectbox("Tipo di Filtro", o['filter'],
                     index=_idx(o['filter'], d['filter_type']), key="comp_filter_type")
        st.slider("Crop Ratio", 0.75, 0.98,
                  value=d['crop_ratio'], step=0.01, key="comp_crop_ratio")
        st.selectbox("Border Mode", o['border'],
                     index=_idx(o['border'], d['border_mode']), key="comp_border_mode")

    if "block_matching" in selected_methods:
        with st.expander("📦 Parametri Block Matching", expanded=False):
            st.select_slider("Block Size (px)", options=[8, 16, 32, 64],
                             value=d['block_size'], key="comp_block_size")
            st.slider("Search Range (px)", 4, 32,
                      value=d['search_range'], step=2, key="comp_search_range")
            st.selectbox("Metrica di Matching", o['metric'],
                         index=_idx(o['metric'], d['metric']), key="comp_metric")
            st.selectbox("Metodo di Aggregazione", o['aggregation'],
                         index=_idx(o['aggregation'], d['aggregation']), key="comp_aggregation")
            st.checkbox("Stima Rotazione", value=d['estimate_rotation'],
                        key="comp_estimate_rotation_bm")

    if any(m.startswith("optical_shi_tomasi") for m in selected_methods):
        with st.expander("🌊 Parametri Optical Flow - Shi-Tomasi", expanded=False):
            st.slider("Max Corners", 200, 2000,
                      value=d['max_corners'], step=100, key="comp_max_corners")
            st.slider("Quality Level", 0.001, 0.01,
                      value=d['quality_level'], step=0.001, format="%.3f", key="comp_quality_level")
            st.slider("Min Distance", 5, 20,
                      value=d['min_distance'], key="comp_min_distance")
            st.select_slider("Window Size", options=[11, 15, 21, 25, 31],
                             value=d['win_size'], key="comp_win_size")
            st.slider("RANSAC Threshold", 1.0, 15.0,
                      value=d['ransac_threshold_st'], step=0.5, key="comp_ransac_threshold_st")

    if any(m.startswith("optical_orb") for m in selected_methods):
        with st.expander("🎯 Parametri ORB Matching", expanded=False):
            st.slider("N° Features", 200, 1000,
                      value=d['orb_n_features'], step=50, key="comp_orb_n_features")
            st.slider("Scale Factor", 1.1, 1.5,
                      value=d['orb_scale_factor'], step=0.05, key="comp_orb_scale_factor")
            st.slider("Pyramid Levels", 4, 12,
                      value=d['orb_n_levels'], key="comp_orb_n_levels")
            st.slider("Ratio Threshold", 0.5, 0.9,
                      value=d['orb_ratio_threshold'], step=0.05, key="comp_orb_ratio_threshold")
            st.slider("RANSAC Threshold", 1.0, 15.0,
                      value=d['ransac_threshold_orb'], step=0.5, key="comp_ransac_threshold_orb")


# ──────────────────────────────────────────────────────────────
# Sezione di processo (progress + run)
# ──────────────────────────────────────────────────────────────

def _build_progress_ui(n_methods):
    """Crea i placeholder per la barra di progresso. Restituisce (elements_dict, slots_list)."""
    progress_title   = st.empty()
    overall_progress = st.empty()
    prog_l, prog_r   = st.columns(2)
    with prog_l:
        slots_left  = [st.empty() for _ in range(4)]
    with prog_r:
        slots_right = [st.empty() for _ in range(3)]
    algo_slots = slots_left + slots_right
    return progress_title, overall_progress, algo_slots


def run_comparison_with_progress(selected_methods, input_path, configs):
    """Avvia il loop di stabilizzazione con UI di progresso integrata."""
    
    progress_title, overall_progress, algo_slots = _build_progress_ui(len(selected_methods))

    params = read_params_from_session(st.session_state, configs['defaults'])

    def _overall(frac):
        overall_progress.progress(frac)

    def _algo_slot(idx, frac, text):
        algo_slots[idx].progress(frac, text=text)

    def _title(text):
        progress_title.text(text)

    callbacks = {
        'overall':   _overall,
        'algo_slot': _algo_slot,
        'title':     _title,
    }

    results = run_all_methods(selected_methods, input_path, params, callbacks)
    st.session_state['comparison_results'] = results
    st.success(f"🎉 Elaborati {len(selected_methods)} metodi!")
    return results


# ──────────────────────────────────────────────────────────────
# Risultati: video combinato + video individuali
# ──────────────────────────────────────────────────────────────

def render_combined_video_section(successful_results, input_path):
    """Sezione per la generazione e visualizzazione del video comparativo in griglia."""
    st.markdown("#### 🎬 Video Comparativo Sincronizzato")
    st.info("💡 Genera un singolo video con tutti i metodi side by side")

    if st.button("🔄 Genera Video Comparativo", type="primary", key="generate_combined"):
        with st.spinner("⏳ Creazione video comparativo in griglia..."):
            video_list = [input_path]
            titles     = ['Originale']

            for method_name, result in successful_results.items():
                video_list.append(result['display_path'])
                titles.append(method_name.replace('_', ' ').title())

            combined_path = combine_videos_grid(video_list, titles)

            if combined_path and Path(combined_path).exists():
                web_combined = combined_path.replace('.mp4', '_web.mp4')
                ok = convert_to_web_compatible(combined_path, web_combined)
                st.session_state['combined_video_path']    = combined_path
                st.session_state['combined_video_display'] = web_combined if ok else combined_path
                st.success("✅ Video comparativo generato con successo!")
            else:
                st.error("❌ Errore nella generazione del video comparativo")

    if 'combined_video_display' in st.session_state:
        st.markdown("##### 🎥 Video Comparativo con Tutti i Metodi")
        st.video(st.session_state['combined_video_display'])
        if 'combined_video_path' in st.session_state:
            with open(st.session_state['combined_video_path'], 'rb') as f:
                st.download_button(
                    label="⬇️ Scarica Video Comparativo",
                    data=f,
                    file_name="comparison_grid.mp4",
                    mime="video/mp4",
                    key="download_combined",
                )


def render_individual_videos(successful_results, input_path):
    """Mostra i video stabilizzati in griglia (3 colonne), incluso l'originale."""
    st.markdown("#### 🎬 Video Stabilizzati")

    num_cols = min(3, len(successful_results) + 1)
    cols     = st.columns(num_cols)

    with cols[0]:
        st.markdown("**📹 Video Originale (Riferimento)**")
        st.video(input_path)
        st.caption("")

    result_items = list(successful_results.items())
    col_offset   = 1

    for idx, (method_name, result) in enumerate(result_items):
        col_idx = (idx + col_offset) % num_cols
        if col_idx == 0 and idx > 0:
            cols = st.columns(num_cols)
        with cols[col_idx]:
            st.markdown(f"**{method_name.replace('_', ' ').title()}**")
            st.video(result['display_path'])
            st.caption(f"⏱️ {result['processing_time']:.2f}s")


def render_results_section(successful_results, input_path):
    """Raggruppa la sezione dei risultati (video comparativo + individuali)."""
    if not successful_results:
        st.error("❌ Nessun metodo ha prodotto risultati validi")
        return

    st.markdown("---")
    render_combined_video_section(successful_results, input_path)
    st.markdown("---")
    render_individual_videos(successful_results, input_path)


# ──────────────────────────────────────────────────────────────
# Entry point del tab
# ──────────────────────────────────────────────────────────────

def render_tab_process(configs):
    """
    Renderizza Tab 1 completo.

    Args:
        configs: output di config_loader.load_configs()
    """
    if 'input_path' not in st.session_state:
        st.info("📹 Carica prima un video dalla sidebar")
        return

    selected_methods = render_method_selection()

    if not selected_methods:
        st.warning("⚠️ Seleziona almeno un metodo da confrontare")
        return

    st.success(f"✅ {len(selected_methods)} metodi selezionati")
    st.markdown("---")

    conf_col, progress_col = st.columns([1, 2])

    with conf_col:
        render_params_section(selected_methods, configs)
        st.markdown("")
        _, btn_mid, _ = st.columns([0.2, 4, 0.2])
        with btn_mid:
            avvia_confronto = st.button(
                "▶️ Avvia Confronto",
                type="primary",
                key="compare_button",
                width='stretch',
            )

    with progress_col:
        if avvia_confronto:
            run_comparison_with_progress(
                selected_methods,
                st.session_state['input_path'],
                configs,
            )

    if 'comparison_results' in st.session_state:
        results = st.session_state['comparison_results']
        successful = {k: v for k, v in results.items() if v.get('success', False)}
        render_results_section(successful, st.session_state['input_path'])


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _idx(lst, value):
    """Indice di un valore in una lista, 0 se non trovato."""
    try:
        return lst.index(value)
    except ValueError:
        return 0
