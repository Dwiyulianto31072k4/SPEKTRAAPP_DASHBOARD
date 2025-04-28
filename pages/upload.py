import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

def show_upload_page():
    """
    Fungsi untuk menampilkan halaman upload dan preprocessing
    yang disederhanakan dengan kode standar yang telah ditentukan
    """
    st.markdown('<p class="section-title">Upload Customer Data</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Excel file with customer data", type=["xlsx", "xls"])

    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name

        try:
            # Baca data
            data = pd.read_excel(uploaded_file)
            initial_rows = len(data)
            st.success(f"File '{uploaded_file.name}' successfully loaded with {initial_rows} rows and {data.shape[1]} columns!")
            
            # Tampilkan preview data
            st.markdown('<p class="section-title">Data Preview (Before Processing)</p>', unsafe_allow_html=True)
            st.dataframe(data.head())
            
            # Tombol untuk preprocessing
            if st.button("Preprocess Data", key="preprocess_button"):
                with st.spinner("Preprocessing data..."):
                    # Terapkan preprocessing standar
                    processed_data = apply_standard_preprocessing(data)
                    
                    # Tampilkan hasil
                    st.success("✅ Data preprocessing completed successfully!")
                    st.markdown('<p class="section-title">Processed Data Preview</p>', unsafe_allow_html=True)
                    st.dataframe(processed_data.head())
                    
                    # Tampilkan informasi tipe data
                    with st.expander("Data Type Information"):
                        st.write(processed_data.dtypes)
                    
                    # Simpan ke session state
                    st.session_state.data = processed_data
                    
                    # Pastikan folder temp ada
                    if not os.path.exists("temp"):
                        os.makedirs("temp")
                    
                    # Simpan file
                    processed_data.to_excel("temp/processed_data.xlsx", index=False)
                    st.session_state.eda_completed = True
                    
                    # Tampilkan next steps
                    st.markdown("### Next Steps")
                    st.success("You can now proceed to the Exploratory Data Analysis section!")
                    
                    # Tombol langsung ke EDA
                    if st.button("Go to EDA"):
                        st.session_state.page = "Exploratory Data Analysis"
                        st.experimental_rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.warning("Please check your file and try again.")
    
    else:
        st.write("Or use our example data to explore the application:")
        if st.button("Use Example Data"):
            example_data = create_example_data()
            
            # Terapkan preprocessing standar
            example_data = apply_standard_preprocessing(example_data)
            
            st.success("✅ Example data loaded successfully!")
            st.session_state.data = example_data
            st.session_state.uploaded_file_name = "example_data.xlsx"
            
            # Buat folder temp jika belum ada
            if not os.path.exists("temp"):
                os.makedirs("temp")
            
            # Simpan contoh data ke file
            example_data.to_excel("temp/processed_data.xlsx", index=False)
            st.dataframe(example_data.head())
            st.session_state.eda_completed = True
            
            # Tampilkan next steps
            st.markdown("### Next Steps")
            st.success("You can now proceed to the Exploratory Data Analysis section!")
            
            # Tombol langsung ke EDA
            if st.button("Go to EDA"):
                st.session_state.page = "Exploratory Data Analysis"
                st.experimental_rerun()

def apply_standard_preprocessing(data):
    """
    Menerapkan standar preprocessing sesuai kode yang ditentukan
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data mentah yang akan diproses
    
    Returns:
    --------
    pandas.DataFrame
        Data yang telah diproses dengan standar yang ditentukan
    """
    # Buat salinan data untuk diproses
    processed_data = data.copy()
    
    # 1. Konversi kolom tanggal dengan format standar
    try:
        processed_data['FIRST_PPC_DATE'] = pd.to_datetime(processed_data['FIRST_PPC_DATE'], format='%Y%m%d', errors='coerce')
        processed_data['FIRST_MPF_DATE'] = pd.to_datetime(processed_data['FIRST_MPF_DATE'], format='%Y%m%d', errors='coerce')
        processed_data['LAST_MPF_DATE'] = pd.to_datetime(processed_data['LAST_MPF_DATE'], format='%Y%m%d', errors='coerce')
        processed_data['CONTRACT_ACTIVE_DATE'] = pd.to_datetime(processed_data['CONTRACT_ACTIVE_DATE'], errors='coerce')
        processed_data['BIRTH_DATE'] = pd.to_datetime(processed_data['BIRTH_DATE'], errors='coerce')
        
        # Tambahkan kolom YearMonth dari CONTRACT_ACTIVE_DATE
        if 'CONTRACT_ACTIVE_DATE' in processed_data.columns:
            processed_data['YearMonth'] = processed_data['CONTRACT_ACTIVE_DATE'].dt.to_period('M')
    except Exception as e:
        st.warning(f"Warning during date conversion: {e}")
    
    # 2. Hitung usia berdasarkan tahun 2024
    try:
        if 'BIRTH_DATE' in processed_data.columns:
            processed_data['Usia'] = 2024 - processed_data['BIRTH_DATE'].dt.year
    except Exception as e:
        st.warning(f"Warning during age calculation: {e}")
    
    # 3. Konversi tipe data untuk kolom numerik
    try:
        if 'OCPT_CODE' in processed_data.columns:
            processed_data['OCPT_CODE'] = processed_data['OCPT_CODE'].astype('Int64')
        if 'NO_OF_DEPEND' in processed_data.columns:
            processed_data['NO_OF_DEPEND'] = processed_data['NO_OF_DEPEND'].astype('Int64')
    except Exception as e:
        st.warning(f"Warning during integer conversion: {e}")
    
    # 4. Hapus kolom yang tidak diperlukan
    try:
        if 'JMH_CON_NON_MPF' in processed_data.columns:
            processed_data.drop(columns=['JMH_CON_NON_MPF'], inplace=True)
    except Exception as e:
        st.warning(f"Warning when dropping columns: {e}")
    
    # 5. Isi nilai yang hilang di kolom numerik
    try:
        numeric_fill_cols = ['MONTH_INST', 'OCPT_CODE', 'NO_OF_DEPEND', 'Usia']
        for col in numeric_fill_cols:
            if col in processed_data.columns:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
    except Exception as e:
        st.warning(f"Warning during numeric imputation: {e}")
    
    # 6. Isi nilai yang hilang di kolom kategori
    try:
        categorical_fill_cols = ['CUST_SEX', 'EDU_TYPE', 'HOUSE_STAT', 'MARITAL_STAT', 'AREA']
        for col in categorical_fill_cols:
            if col in processed_data.columns:
                if not processed_data[col].empty:
                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
    except Exception as e:
        st.warning(f"Warning during categorical imputation: {e}")
    
    # 7. Pastikan semua kolom tanggal dalam format datetime
    try:
        date_cols = ['FIRST_PPC_DATE', 'FIRST_MPF_DATE', 'LAST_MPF_DATE', 'CONTRACT_ACTIVE_DATE', 'BIRTH_DATE']
        for col in date_cols:
            if col in processed_data.columns:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
    except Exception as e:
        st.warning(f"Warning during final date conversion: {e}")
    
    # 8. Pastikan kolom numerik bertipe float
    try:
        numerical_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MONTH_INST', 'Usia']
        for col in numerical_cols:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
    except Exception as e:
        st.warning(f"Warning during numeric conversion: {e}")
    
    # 9. Bersihkan whitespace di kolom kategorikal
    try:
        categorical_cols = ['MPF_CATEGORIES_TAKEN', 'CUST_SEX', 'EDU_TYPE', 'HOUSE_STAT', 'MARITAL_STAT', 'AREA']
        for col in categorical_cols:
            if col in processed_data.columns and processed_data[col].dtype == "object":
                processed_data[col] = processed_data[col].str.strip()
    except Exception as e:
        st.warning(f"Warning during string cleaning: {e}")
    
    return processed_data

# Menggunakan contoh data yang sudah ada
def create_example_data():
    """Create a sample customer dataset for testing and demonstration purposes."""
    from utils.data_utils import create_example_data as create_data
    return create_data()
