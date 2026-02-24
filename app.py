"""
Streamlit App per Video Stabilization System
Entry point – tutta la logica applicativa è in ui/
"""

import streamlit as st

from ui.styles import inject_css
from ui.config_loader import load_configs
from ui.video_utils import get_video_info
from ui.tab_1v1 import render_tab_1v1
from ui.tab_process import render_tab_process
from ui.tab_metrics import render_tab_metrics

import tempfile

st.set_page_config(
    page_title="Video Stabilization App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)



def _render_sidebar():
    """Sidebar: upload video."""
    st.sidebar.subheader("📹 Video Input")

    uploaded = st.sidebar.file_uploader(
        "Carica un video da stabilizzare",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supportati: MP4, AVI, MOV, MKV",
    )

    if uploaded is not None:
        # Evita di ricreare il temp file ad ogni re-render di Streamlit:
        # confronta nome+dimensione per rilevare un cambio di file.
        file_id = (uploaded.name, uploaded.size)
        if st.session_state.get('_uploaded_file_id') != file_id:
            uploaded.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded.read())
                input_path = tmp.name
            info = get_video_info(input_path)
            if info:
                st.session_state['_uploaded_file_id'] = file_id
                st.session_state['input_path'] = input_path
                st.session_state['video_info'] = info
            else:
                st.sidebar.error("❌ Errore nel caricamento")
                return

        if 'video_info' in st.session_state:
            info = st.session_state['video_info']
            st.sidebar.success("✅ Video caricato!")
            st.sidebar.caption(
                f"📐 {info['width']}x{info['height']} | "
                f"🎞️ {info['frame_count']} frame | "
                f"⏱️ {info['duration']:.1f}s"
            )
    else:
        # File rimosso: resetta lo stato
        st.session_state.pop('_uploaded_file_id', None)


def main():
    inject_css()
    configs = load_configs()
    _render_sidebar()

    tab_1v1, tab_process, tab_metrics = st.tabs([
        "🔍 Stabilizza",
        "📤 Confronta",
        "📊 Analisi confronto",
    ])

    with tab_1v1:
        render_tab_1v1(configs)

    with tab_process:
        render_tab_process(configs)

    with tab_metrics:
        render_tab_metrics()


if __name__ == "__main__":
    main()

