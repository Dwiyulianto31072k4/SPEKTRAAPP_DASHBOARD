import streamlit as st
import os
from datetime import datetime

# Import modul halaman
from pages.upload import show_upload_page
from pages.eda import show_eda_page
from pages.segmentation import show_segmentation_page
from pages.promo_mapping import show_promo_mapping_page
from pages.dashboard import show_dashboard_page
from pages.export import show_export_page

# Set konfigurasi halaman
st.set_page_config(
    page_title="SPEKTRA Customer Segmentation SPEKTRA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sembunyikan navigasi default Streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .css-k0sv6k {display: none;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
            .main > div:first-child {
                padding-top: 0rem;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load CSS dari file
def load_css():
    with open("styles/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create a directory for temporary files if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Load logo dan tampilkan di sidebar
import base64

def load_sidebar():
    try:
        # Baca gambar dan encode ke base64
        with open("assets/logo.png", "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

        # Tampilkan logo di tengah dan full width (maks 180px)
        st.sidebar.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" style="width: 100%; max-width: 180px; margin-bottom: 10px;" />
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        st.sidebar.title("SPEKTRA")
    
    # Judul sidebar
    st.sidebar.markdown('<p class="sidebar-title">Customer Segmentation & Promo Mapping</p>', unsafe_allow_html=True)

    # Navigasi
    st.sidebar.markdown("### Navigation")
    pages = ["Upload & Preprocessing", "Exploratory Data Analysis", "Segmentation Analysis", 
             "Promo Mapping", "Dashboard", "Export & Documentation"]
    selected_page = st.sidebar.radio("Go to", pages)
    
    # Info tambahan di sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Aplikasi ini** adalah aplikasi segmentasi pelanggan dan pemetaan promo.
    
    Dibangun oleh Tim Data Science FIFGROUP.
    """)
    
    return selected_page


# Inisialisasi session state jika belum ada
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'segmented_data' not in st.session_state:
        st.session_state.segmented_data = None
    if 'promo_mapped_data' not in st.session_state:
        st.session_state.promo_mapped_data = None
    if 'rfm_data' not in st.session_state:
        st.session_state.rfm_data = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'eda_completed' not in st.session_state:
        st.session_state.eda_completed = False
    if 'segmentation_completed' not in st.session_state:
        st.session_state.segmentation_completed = False
    if 'promo_mapping_completed' not in st.session_state:
        st.session_state.promo_mapping_completed = False

# Main function
def main():
    try:
        # Load CSS
        load_css()
        
        # Inisialisasi session state
        init_session_state()
        
        # Load sidebar dan dapatkan halaman yang dipilih
        selected_page = load_sidebar()
        
        # Tampilkan judul utama
        st.markdown('<p class="main-title">SPEKTRA Customer Segmentation & Promo Mapping</p>', unsafe_allow_html=True)
        
        # Tampilkan halaman sesuai pilihan
        if selected_page == "Upload & Preprocessing":
            show_upload_page()
        elif selected_page == "Exploratory Data Analysis":
            show_eda_page()
        elif selected_page == "Segmentation Analysis":
            show_segmentation_page()
        elif selected_page == "Promo Mapping":
            show_promo_mapping_page()
        elif selected_page == "Dashboard":
            show_dashboard_page()
        elif selected_page == "Export & Documentation":
            show_export_page()
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>Developed with ❤️ by the Data Science Team at FIFGROUP · Powered by Streamlit</p>
            <p>Last updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please refresh the page and try again.")

# Run the app
if __name__ == "__main__":
    main()
