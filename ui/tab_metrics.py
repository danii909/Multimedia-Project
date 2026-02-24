"""Tab 2: Metriche & Analisi – tabelle, traiettorie, grafici comparativi e download"""
import json
import streamlit as st
import pandas as pd
from ui.plot_utils import (
    create_trajectory_plot,
    create_rms_comparison_chart,
    create_jitter_comparison_chart,
)


def _render_comparison_metrics(successful_results):
    """Tabella riepilogativa + grafici comparativi tra metodi."""

    # Avviso globale se almeno un metodo ha avuto la window clampata
    clamped = [
        (name, r['metrics'].get('smoothing_window_requested'), r['metrics'].get('smoothing_window_used'))
        for name, r in successful_results.items()
        if r['metrics'].get('smoothing_window_clamped')
    ]
    if clamped:
        first_name, sw_req, sw_used = clamped[0]
        n_frames = successful_results[first_name]['metrics'].get('num_frames', '?')
        st.warning(
            f"⚠️ Smoothing window ridotta automaticamente da **{sw_req}** a **{sw_used}** frame "
            f"(il video ha {n_frames} frame). "
            f"I valori di jitter reduction sono calcolati sulla window effettiva."
        )

    # Tabella
    st.markdown("#### 📋 Tabella Riepilogativa")
    rows = []
    for method_name, result in successful_results.items():
        m = result['metrics']
        rows.append({
            'Metodo':           method_name.replace('_', ' ').title(),
            'RMS X (px)':       f"{m.get('rms_dx', 0):.2f}",
            'RMS Y (px)':       f"{m.get('rms_dy', 0):.2f}",
            'RMS Angle (°)':    f"{m.get('rms_angle', 0):.3f}",
            'Jitter Red. X (%)': f"{m.get('jitter_reduction_x', 0):.1f}",
            'Jitter Red. Y (%)': f"{m.get('jitter_reduction_y', 0):.1f}",
            'Tempo (s)':        f"{result['processing_time']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch')

    st.markdown("---")

    # Traiettorie per metodo – raggruppate per famiglia algoritmo
    st.markdown("#### 📉 Traiettorie - Confronto per Metodo")

    _GROUPS = [
        ("📦 Block Matching",          lambda k: k.startswith("block")),
        ("🌊 Optical Flow (LK)",        lambda k: k.startswith("optical_shi_tomasi")),
        ("🎯 ORB Matching",             lambda k: k.startswith("optical_orb")),
    ]

    for group_label, group_filter in _GROUPS:
        group_items = [(k, v) for k, v in successful_results.items() if group_filter(k)]
        if not group_items:
            continue
        st.markdown(f"##### {group_label}")
        cols = st.columns(len(group_items))
        for col_idx, (method_name, result) in enumerate(group_items):
            with cols[col_idx]:
                st.markdown(f"**{method_name.replace('_', ' ').title()}**")
                fig = create_trajectory_plot(result['metrics'])
                if fig:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Dati traiettoria non disponibili")
                m1, m2, m3 = st.columns(3)
                m = result['metrics']
                m1.metric("RMS X",  f"{m.get('rms_dx', 0):.2f} px")
                m2.metric("RMS Y",  f"{m.get('rms_dy', 0):.2f} px")
                m3.metric("Tempo",  f"{result['processing_time']:.2f}s")
        st.markdown("")

    st.markdown("---")

    # Grafici comparativi
    st.markdown("#### 📊 Grafici Comparativi")
    st.pyplot(create_rms_comparison_chart(successful_results))
    st.pyplot(create_jitter_comparison_chart(successful_results))

    st.markdown("---")

    # Download
    st.markdown("#### 💾 Download")
    export = {
        method: {
            'metrics':          result['metrics'],
            'processing_time':  result['processing_time'],
        }
        for method, result in successful_results.items()
    }
    st.download_button(
        label="📥 Scarica Metriche Comparative (JSON)",
        data=json.dumps(export, indent=2),
        file_name="comparison_metrics.json",
        mime="application/json",
    )


def _render_single_metrics(metrics, processing_time):
    """Metriche e grafici per un singolo metodo."""
    st.markdown("### 📈 Metriche di Stabilità")

    # Avviso se la smoothing window è stata clampata
    if metrics.get('smoothing_window_clamped'):
        sw_req  = metrics.get('smoothing_window_requested', '?')
        sw_used = metrics.get('smoothing_window_used', '?')
        st.warning(
            f"⚠️ Smoothing window ridotta automaticamente da **{sw_req}** a **{sw_used}** frame "
            f"(il video ha {metrics.get('num_frames', '?')} frame). "
            f"I valori di jitter reduction sono calcolati sulla window effettiva."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMS Displacement X", f"{metrics.get('rms_dx', 0):.2f} px",
              help="Root Mean Square dello spostamento orizzontale")
    c2.metric("RMS Displacement Y", f"{metrics.get('rms_dy', 0):.2f} px",
              help="Root Mean Square dello spostamento verticale")
    c3.metric("RMS Rotation",       f"{metrics.get('rms_angle', 0):.3f}°",
              help="Root Mean Square della rotazione")
    c4.metric("Processing Time",    f"{processing_time:.2f}s",
              help="Tempo totale di elaborazione")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    for col, key, label in [
        (c1, 'jitter_reduction_x',     'Jitter Reduction X'),
        (c2, 'jitter_reduction_y',     'Jitter Reduction Y'),
        (c3, 'jitter_reduction_angle', 'Jitter Reduction Angle'),
    ]:
        val = metrics.get(key, 0)
        col.metric(label, f"{val:.1f}%",
                   delta=f"{val:.1f}%" if val > 0 else None,
                   delta_color="normal" if val > 0 else "inverse")

    st.markdown("---")
    st.markdown("### 📉 Visualizzazioni")
    st.markdown("#### Traiettoria Raw vs Smoothed")
    fig = create_trajectory_plot(metrics)
    if fig:
        st.pyplot(fig)
    else:
        st.info("Dati traiettoria non disponibili")

    st.markdown("---")
    with st.expander("🔍 Dettagli Tecnici Completi"):
        st.json(metrics)

    st.download_button(
        label="📥 Scarica Metriche (JSON)",
        data=json.dumps(metrics, indent=2),
        file_name="stabilization_metrics.json",
        mime="application/json",
    )


def render_tab_metrics():
    """Renderizza Tab 2 completo."""
    if 'comparison_results' in st.session_state:
        results     = st.session_state['comparison_results']
        successful  = {k: v for k, v in results.items() if v.get('success', False)}
        if successful:
            _render_comparison_metrics(successful)
        else:
            st.warning("⚠️ Nessun metodo ha prodotto risultati validi")

    elif 'metrics' in st.session_state:
        _render_single_metrics(
            st.session_state['metrics'],
            st.session_state.get('processing_time', 0),
        )

    else:
        st.info("📊 Esegui prima la stabilizzazione per vedere le metriche")
