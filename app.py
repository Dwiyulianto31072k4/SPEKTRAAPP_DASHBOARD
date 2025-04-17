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
    
    # Get current page from session state
    current_page = st.session_state.get('page', "Upload & Preprocessing")
    
    selected_page = st.sidebar.radio("Go to", pages, index=pages.index(current_page))
    
    # Update session state with new page
    if selected_page != current_page:
        st.session_state.page = selected_page
    
    # Info tambahan di sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Aplikasi ini** adalah aplikasi segmentasi pelanggan dan pemetaan promo.
    
    Dibangun oleh Tim Data Science FIFGROUP.
    """)
    
    # Add data cleanliness indicator
    if 'data' in st.session_state and st.session_state.data is not None:
        data_clean = True
        unknown_count = 0
        
        # Check for any NaN or "Unknown" values
        for col in st.session_state.data.columns:
            if st.session_state.data[col].isna().any():
                data_clean = False
                unknown_count += st.session_state.data[col].isna().sum()
            elif st.session_state.data[col].dtype == 'object' or col.endswith('_Kategori'):
                if any(st.session_state.data[col].str.contains('Unknown', case=False, na=False)):
                    data_clean = False
                    unknown_count += st.session_state.data[col].str.contains('Unknown', case=False, na=False).sum()
        
        if data_clean:
            st.sidebar.success("✅ Data Quality: Clean (No unknown/NaN values)")
        else:
            st.sidebar.error(f"⚠️ Data Quality: {unknown_count} unknown/NaN values detected")
    
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
    if 'page' not in st.session_state:
        st.session_state.page = "Upload & Preprocessing"

# Function to validate data cleanliness
def check_data_quality():
    """Check data quality and return True if clean, False otherwise"""
    if 'data' not in st.session_state or st.session_state.data is None:
        return False
    
    data = st.session_state.data
    
    # Check for NaN values in any column
    if data.isna().any().any():
        return False
    
    # Check for "Unknown" values in object columns
    for col in data.columns:
        if data[col].dtype == 'object' or col.endswith('_Kategori'):
            if any(data[col].str.contains('Unknown', case=False, na=False)) or \
               any(data[col].str.contains('nan', case=False, na=False)) or \
               any(data[col].str.contains('None', case=False, na=False)):
                return False
    
    return True

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
        
        # Check data quality if beyond upload page
        if selected_page != "Upload & Preprocessing" and 'data' in st.session_state and st.session_state.data is not None:
            if not check_data_quality():
                st.warning("⚠️ Your data contains unknown or missing values. It's recommended to return to Upload & Preprocessing to clean your data.")
                if st.button("Go to Upload & Preprocessing"):
                    st.session_state.page = "Upload & Preprocessing"
                    st.experimental_rerun()
        
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
