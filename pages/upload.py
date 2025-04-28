import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

def show_upload_page():
    """
    Fungsi untuk menampilkan halaman upload dan preprocessing
    yang disederhanakan dengan kode standar yang ditentukan
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
                    
                    # Tambahkan verifikasi kualitas data
                    has_na = processed_data.isna().any().any()
                    unknown_values = False

                    # Periksa nilai "Unknown" di kolom kategorikal
                    for col in processed_data.select_dtypes(include=['object']).columns:
                        if any(processed_data[col].str.contains('Unknown|NaN|None', case=False, regex=True, na=False)):
                            unknown_values = True
                            break

                    if has_na or unknown_values:
                        st.error("❌ Data masih mengandung nilai yang hilang atau tidak diketahui setelah preprocessing!")
                        st.info("Memperbaiki masalah data secara otomatis...")
                        
                        # Hapus baris dengan nilai yang hilang atau tidak diketahui
                        processed_data = processed_data.dropna()
                        
                        for col in processed_data.select_dtypes(include=['object']).columns:
                            mask = processed_data[col].str.contains('Unknown|NaN|None', case=False, regex=True, na=False)
                            if mask.any():
                                processed_data = processed_data[~mask]
                        
                        if processed_data.empty:
                            st.error("Semua data hilang setelah pembersihan ketat. Harap periksa file sumber Anda.")
                            return
                        
                        st.success(f"✅ Data berhasil dibersihkan! {len(data) - len(processed_data)} baris dihapus.")
                    else:
                        st.success("✅ Data preprocessing completed successfully with 100% clean data!")
                    
                    # Tampilkan hasil
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
    Menerapkan standar preprocessing yang ketat untuk memastikan data benar-benar bersih
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data mentah yang akan diproses
    
    Returns:
    --------
    pandas.DataFrame
        Data yang telah diproses dan benar-benar bersih
    """
    # Buat salinan data untuk diproses
    processed_data = data.copy()
    
    # 1. Konversi kolom tanggal dengan format standar
    date_cols = ['FIRST_PPC_DATE', 'FIRST_MPF_DATE', 'LAST_MPF_DATE', 'CONTRACT_ACTIVE_DATE', 'BIRTH_DATE']
    for col in date_cols:
        if col in processed_data.columns:
            processed_data[col] = pd.to_datetime(processed_data[col], format='%Y%m%d', errors='coerce')
    
    # 2. Hitung usia berdasarkan tahun 2024
    if 'BIRTH_DATE' in processed_data.columns:
        processed_data['Usia'] = 2024 - processed_data['BIRTH_DATE'].dt.year
    
    # 3. Konversi tipe data untuk kolom numerik
    int_cols = ['OCPT_CODE', 'NO_OF_DEPEND']
    for col in int_cols:
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').astype('Int64')
    
    # 4. Hapus kolom yang tidak diperlukan
    drop_cols = ['JMH_CON_NON_MPF']
    for col in drop_cols:
        if col in processed_data.columns:
            processed_data.drop(columns=[col], inplace=True)
    
    # 5. Pastikan kolom numerik bertipe float
    float_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MONTH_INST', 'Usia', 
                 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 'LAST_MPF_AMOUNT', 'LAST_MPF_INST']
    for col in float_cols:
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
    
    # 6. Bersihkan whitespace di kolom kategorikal
    cat_cols = ['MPF_CATEGORIES_TAKEN', 'CUST_SEX', 'EDU_TYPE', 'HOUSE_STAT', 'MARITAL_STAT', 'AREA']
    for col in cat_cols:
        if col in processed_data.columns and processed_data[col].dtype == "object":
            processed_data[col] = processed_data[col].str.strip()
    
    # 7. Tambahkan kolom YearMonth
    if 'CONTRACT_ACTIVE_DATE' in processed_data.columns:
        processed_data['YearMonth'] = processed_data['CONTRACT_ACTIVE_DATE'].dt.to_period('M')
    
    # 8. Isi nilai yang hilang dengan nilai yang masuk akal
    # Untuk kolom numerik: gunakan median
    numeric_cols = float_cols + [col for col in int_cols if col in processed_data.columns]
    for col in numeric_cols:
        if col in processed_data.columns:
            # Isi nilai yang hilang dengan median
            if processed_data[col].notna().any():  # Pastikan ada setidaknya satu nilai non-NA
                median_value = processed_data[col].median()
                processed_data[col].fillna(median_value, inplace=True)
    
    # Untuk kolom kategorikal: gunakan modus
    for col in cat_cols:
        if col in processed_data.columns:
            # Isi nilai yang hilang dengan modus
            if processed_data[col].notna().any():  # Pastikan ada setidaknya satu nilai non-NA
                mode_value = processed_data[col].mode()[0]
                processed_data[col].fillna(mode_value, inplace=True)
    
    # 9. Pastikan tidak ada nilai "Unknown" atau string yang tidak sesuai di kolom kategorikal
    for col in cat_cols:
        if col in processed_data.columns and processed_data[col].dtype == "object":
            # Ganti nilai tidak diketahui dengan modus
            if processed_data[col].notna().any():  # Pastikan ada setidaknya satu nilai non-NA
                mode_value = processed_data[col].mode()[0]
                unknown_mask = processed_data[col].str.contains('Unknown', case=False, na=True) | \
                              processed_data[col].str.contains('NaN', case=False, na=True) | \
                              processed_data[col].str.contains('None', case=False, na=True) | \
                              processed_data[col].isna()
                processed_data.loc[unknown_mask, col] = mode_value
    
    # 10. Usia harus dalam rentang masuk akal, jika tidak dalam rentang, ganti dengan median
    if 'Usia' in processed_data.columns:
        if processed_data['Usia'].notna().any():  # Pastikan ada setidaknya satu nilai non-NA
            median_age = processed_data['Usia'].median()
            age_mask = (processed_data['Usia'] < 18) | (processed_data['Usia'] > 100) | processed_data['Usia'].isna()
            processed_data.loc[age_mask, 'Usia'] = median_age
            
            # Buat kategori usia
            bins = [0, 25, 35, 45, 55, 100]
            labels = ['<25', '25-35', '35-45', '45-55', '55+']
            processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
    
    # 11. Create business metrics
    if 'TOTAL_PRODUCT_MPF' in processed_data.columns:
        processed_data['Multi_Product_Flag'] = (processed_data['TOTAL_PRODUCT_MPF'] > 1).astype(int)
    
    if 'LAST_MPF_DATE' in processed_data.columns:
        reference_date = pd.Timestamp.now()
        processed_data['Recency_Days'] = (reference_date - processed_data['LAST_MPF_DATE']).dt.days
        
        # Recency categories
        processed_data['Recency_Category'] = pd.cut(
            processed_data['Recency_Days'],
            bins=[0, 30, 90, 180, 365, float('inf')],
            labels=['Very Recent', 'Recent', 'Moderate', 'Lapsed', 'Inactive'],
            include_lowest=True
        )
    
    # 12. Satu pemeriksaan akhir untuk NaN dan hapus jika ada
    # Untuk kolom yang paling penting
    important_cols = ['CUST_NO', 'LAST_MPF_DATE', 'TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF']
    for col in important_cols:
        if col in processed_data.columns:
            processed_data = processed_data[processed_data[col].notna()]
    
    # 13. Konversi kolom kategori ke string untuk menghindari masalah
    cat_cols_all = cat_cols + ['Usia_Kategori', 'Recency_Category']
    for col in cat_cols_all:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].astype(str)
    
    return processed_data

def create_example_data():
    """Create a sample customer dataset for testing and demonstration purposes."""
    from utils.data_utils import create_example_data as create_data
    return create_data()
