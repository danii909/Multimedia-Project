"""CSS personalizzato per l'app Streamlit"""
import streamlit as st


def inject_css():
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
