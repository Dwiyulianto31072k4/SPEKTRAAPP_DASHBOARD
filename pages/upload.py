import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re

def show_upload_page():
    """
    Fungsi untuk menampilkan halaman upload dan preprocessing
    dengan validasi data ketat untuk menghilangkan NaN/unknown
    """
    st.markdown('<p class="section-title">Upload Customer Data</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Excel file with customer data", type=["xlsx", "xls"])

    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name

        try:
            # Baca semua kolom sebagai string untuk menjaga konsistensi
            data = pd.read_excel(uploaded_file, dtype=str)
            initial_rows = len(data)
            st.success(f"File '{uploaded_file.name}' successfully loaded with {initial_rows} rows and {data.shape[1]} columns!")
            st.markdown('<p class="section-title">Data Preview (Before Cleaning)</p>', unsafe_allow_html=True)
            st.dataframe(data.head())

            # Deteksi otomatis kolom dengan 'DATE' di namanya
            date_cols = [col for col in data.columns if 'DATE' in col.upper()]
            st.markdown(f"🔍 Auto-detected date columns: `{', '.join(date_cols)}`")

            # Tambahkan pengaturan untuk validasi data
            st.markdown("### Data Validation Settings")
            
            # Identifikasi kolom penting yang harus ada data validnya
            required_cols = st.multiselect(
                "Select required columns (rows with NaN in these columns will be removed):",
                options=data.columns.tolist(),
                default=['CUST_NO', 'LAST_MPF_DATE', 'TOTAL_PRODUCT_MPF', 'TOTAL_AMOUNT_MPF'] if all(col in data.columns for col in ['CUST_NO', 'LAST_MPF_DATE', 'TOTAL_PRODUCT_MPF', 'TOTAL_AMOUNT_MPF']) else []
            )
            
            # Checkbox untuk mewajibkan data usia yang valid
            require_valid_age = st.checkbox("Require valid age data (18-100 years old)", value=True)
            
            # Slider untuk menentukan minimum persentase baris valid yang dipertahankan
            min_valid_rows = st.slider(
                "Minimum percentage of valid rows required:",
                min_value=1, max_value=100, value=50,
                help="If the percentage of valid rows falls below this threshold, preprocessing will fail."
            )
            
            # ALWAYS enable strict validation
            strict_validation = True
            st.info("✅ Strict validation is enabled: All unknown/null values will be removed to ensure clean data.")

            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    # Panggil fungsi preprocess_data
                    processed_data = preprocess_data(data, date_cols)
                    
                    if processed_data is None or processed_data.empty:
                        st.error("❌ Preprocessing failed: No valid data rows after filtering.")
                        return
                    
                    # Validasi tambahan berdasarkan pengaturan pengguna
                    valid_rows_mask = np.ones(len(processed_data), dtype=bool)
                    
                    # Periksa kolom yang diperlukan
                    for col in required_cols:
                        if col in processed_data.columns:
                            valid_rows_mask = valid_rows_mask & processed_data[col].notna()
                    
                    # Periksa usia jika diperlukan
                    if require_valid_age and 'Usia' in processed_data.columns:
                        valid_age_mask = (processed_data['Usia'] >= 18) & (processed_data['Usia'] <= 100)
                        valid_rows_mask = valid_rows_mask & valid_age_mask
                    
                    # Apply strict validation - hapus semua NaN dan "Unknown"
                    for col in processed_data.columns:
                        if col != 'CUST_NO':  # Tetap jaga customer ID
                            # Hapus baris dengan NaN
                            valid_rows_mask = valid_rows_mask & processed_data[col].notna()
                            
                            # Hapus baris dengan nilai 'Unknown'/'unknown'/'nan' jika kolom kategorikal
                            if processed_data[col].dtype == 'object' or col.endswith('_Kategori'):
                                unknown_mask = ~(processed_data[col].str.contains('Unknown', case=False, na=False) | 
                                                processed_data[col].str.contains('nan', case=False, na=False) |
                                                processed_data[col].str.contains('NaN', case=False, na=False) |
                                                processed_data[col].str.contains('None', case=False, na=False))
                                valid_rows_mask = valid_rows_mask & unknown_mask
                    
                    # Filter data berdasarkan mask
                    final_data = processed_data[valid_rows_mask].copy()
                    
                    # Periksa persentase baris valid
                    valid_percentage = (len(final_data) / len(data)) * 100
                    st.info(f"Valid data rows: {len(final_data)} out of {initial_rows} ({valid_percentage:.1f}%)")
                    
                    if valid_percentage < min_valid_rows:
                        st.error(f"❌ Preprocessing failed: Only {valid_percentage:.1f}% valid rows (minimum required: {min_valid_rows}%)")
                        return
                    
                    # Final verification untuk memastikan benar-benar tidak ada nilai "Unknown" atau NaN
                    has_unknown = False
                    unknown_cols = []
                    
                    for col in final_data.columns:
                        # Cek NaN values
                        if final_data[col].isna().any():
                            has_unknown = True
                            unknown_cols.append(f"{col} (has NaN)")
                        
                        # Cek string 'Unknown' untuk kolom kategori/objek
                        if final_data[col].dtype == 'object' or col.endswith('_Kategori'):
                            if any(final_data[col].str.contains('Unknown', case=False, na=False)) or \
                               any(final_data[col].str.contains('nan', case=False, na=False)) or \
                               any(final_data[col].str.contains('NaN', case=False, na=False)) or \
                               any(final_data[col].str.contains('None', case=False, na=False)):
                                has_unknown = True
                                unknown_cols.append(f"{col} (has 'Unknown'/'nan'/'None')")
                    
                    if has_unknown:
                        st.warning(f"⚠️ Found remaining unknown values in columns: {', '.join(unknown_cols)}")
                        st.warning("Applying one final cleaning pass...")
                        
                        # Apply one more filtering pass
                        final_clean_mask = np.ones(len(final_data), dtype=bool)
                        
                        for col in final_data.columns:
                            if col != 'CUST_NO':
                                # Remove rows with NaN
                                final_clean_mask = final_clean_mask & final_data[col].notna()
                                
                                # Remove rows with 'Unknown'/'nan'/'None' in categorical columns
                                if final_data[col].dtype == 'object' or col.endswith('_Kategori'):
                                    clean_mask = ~(final_data[col].str.contains('Unknown', case=False, na=False) | 
                                                  final_data[col].str.contains('nan', case=False, na=False) |
                                                  final_data[col].str.contains('NaN', case=False, na=False) |
                                                  final_data[col].str.contains('None', case=False, na=False))
                                    final_clean_mask = final_clean_mask & clean_mask
                        
                        # Apply final filter
                        final_data = final_data[final_clean_mask].copy()
                        st.info(f"After final cleaning: {len(final_data)} valid rows remain")
                        
                        # Check again to be absolutely sure
                        still_has_unknown = False
                        for col in final_data.columns:
                            if final_data[col].isna().any():
                                still_has_unknown = True
                            elif (final_data[col].dtype == 'object' or col.endswith('_Kategori')) and \
                                 (any(final_data[col].str.contains('Unknown', case=False, na=False)) or
                                  any(final_data[col].str.contains('nan', case=False, na=False)) or
                                  any(final_data[col].str.contains('NaN', case=False, na=False)) or
                                  any(final_data[col].str.contains('None', case=False, na=False))):
                                still_has_unknown = True
                        
                        if still_has_unknown:
                            st.error("❌ Unable to completely clean the data. Please check your source data.")
                            return
                    
                    # Tampilkan hasil preprocessing
                    st.success("✅ Data preprocessing completed with 100% clean data!")
                    st.markdown('<p class="section-title">Cleaned Data Preview</p>', unsafe_allow_html=True)
                    st.dataframe(final_data.head())

                    # Tampilkan informasi kolom Usia jika ada
                    if 'Usia' in final_data.columns:
                        st.markdown("### Age Information")
                        st.write(f"All age values are valid (between 18-100 years)")
                        
                        age_stats = final_data['Usia'].describe()
                        st.write(f"Age statistics: Min={age_stats['min']:.1f}, Max={age_stats['max']:.1f}, Mean={age_stats['mean']:.1f}")

                        if 'Usia_Kategori' in final_data.columns:
                            st.write("Age category distribution:")
                            st.write(final_data['Usia_Kategori'].value_counts())
                    
                    # Simpan data yang telah divalidasi ke session state
                    st.session_state.data = final_data

                    # Pastikan folder temp ada
                    if not os.path.exists("temp"):
                        os.makedirs("temp")

                    # Simpan file
                    final_data.to_excel("temp/processed_data.xlsx", index=False)
                    st.session_state.eda_completed = True

                    # Tampilkan next steps
                    st.markdown("### Next Steps")
                    st.success("You can now proceed to the Exploratory Data Analysis section with completely clean data!")
                    
                    # Tambahkan tombol langsung ke EDA
                    if st.button("Go to EDA"):
                        # Switch to EDA page
                        st.session_state.page = "Exploratory Data Analysis"
                        st.experimental_rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.warning("Please check your file and try again.")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.write("Or use our example data to explore the application:")
        if st.button("Use Example Data"):
            example_data = create_example_data()
            
            # Pastikan contoh data benar-benar bersih
            example_data = example_data.dropna()
            
            # Pastikan ada kolom Usia dan kategori
            if 'BIRTH_DATE' in example_data.columns:
                current_date = pd.Timestamp.now()
                example_data['Usia'] = ((current_date - pd.to_datetime(example_data['BIRTH_DATE'])).dt.days / 365.25).round()
                
                # Buat kategori usia
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                example_data['Usia_Kategori'] = pd.cut(example_data['Usia'], bins=bins, labels=labels, right=False)
                example_data['Usia_Kategori'] = example_data['Usia_Kategori'].astype(str)
                
                # Pastikan tidak ada NaN atau nilai yang tidak valid
                example_data = example_data.dropna()
            
            st.success("✅ Example data loaded successfully with 100% clean data!")
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
            st.success("You can now proceed to the Exploratory Data Analysis section with clean example data!")

def clean_date_string(date_string):
    """
    Membersihkan dan menstandardisasi string tanggal
    
    Parameters:
    -----------
    date_string : str
        String tanggal yang akan dibersihkan
    
    Returns:
    --------
    str
        String tanggal yang sudah dibersihkan
    """
    if pd.isna(date_string) or date_string is None or date_string == '':
        return date_string

    # Hapus waktu (contoh: "00.00.00" atau komponen waktu lain)
    date_string = str(date_string)
    date_string = re.sub(r'\s*\d{1,2}[:.]\d{1,2}[:.]\d{1,2}.*$', '', date_string)
    date_string = re.sub(r'\s+00\.00\.00.*$', '', date_string)

    # Bersihkan karakter pemisah yang tidak standar
    date_string = date_string.replace('/', '-').replace('.', '-')
    return date_string.strip()

def preprocess_data(data, date_cols):
    """
    Fungsi untuk memproses data sebelum analisis dengan validasi ketat untuk menghilangkan data NaN/unknown
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data mentah yang akan diproses
    date_cols : list
        Daftar kolom yang berisi tanggal
    
    Returns:
    --------
    pandas.DataFrame
        Data yang telah diproses dan divalidasi (tidak ada NaN)
    """
    import pandas as pd
    import numpy as np
    import datetime
    import re
    
    processed_data = data.copy()
    valid_data_mask = np.ones(len(processed_data), dtype=bool)  # Mask untuk menyimpan baris valid
    
    # Konversi kolom tanggal
    for col in date_cols:
        if col in processed_data.columns:
            # Cek apakah kolom tersebut merupakan tipe numerik (mis. Excel serial number)
            if pd.api.types.is_numeric_dtype(processed_data[col]) or processed_data[col].str.isnumeric().all():
                try:
                    processed_data[col] = pd.to_datetime(processed_data[col], origin='1899-12-30', unit='d', errors='coerce')
                    st.write(f"Converted {col} from numeric (Excel serial) to datetime: {processed_data[col].notna().sum()} valid dates")
                except Exception as e:
                    st.warning(f"Numeric conversion failed for {col} with error: {e}")

            # Jika bukan numeric, lakukan clean-up terlebih dahulu
            original_values = processed_data[col].copy()
            processed_data[col] = processed_data[col].apply(clean_date_string)

            # Coba beberapa format tanggal umum
            date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d', '%d/%m/%Y', '%d.%m.%Y', '%Y.%m.%d']
            converted = False

            for fmt in date_formats:
                try:
                    temp_dates = pd.to_datetime(processed_data[col], format=fmt, errors='coerce')
                    success_rate = temp_dates.notna().sum() / len(processed_data)
                    if success_rate > 0.3:
                        processed_data[col] = temp_dates
                        st.write(f"Converted {col} using format {fmt}: {temp_dates.notna().sum()} valid dates ({success_rate:.1%})")
                        converted = True
                        break
                except Exception:
                    continue

            if not converted:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                success_rate = processed_data[col].notna().sum() / len(processed_data)
                st.write(f"Converted {col} using pandas automatic detection: {processed_data[col].notna().sum()} valid dates ({success_rate:.1%})")
            
            # Update mask untuk validasi tanggal
            valid_data_mask = valid_data_mask & processed_data[col].notna()

    # Khusus untuk BIRTH_DATE dengan penanganan lebih agresif
    if 'BIRTH_DATE' in processed_data.columns:
        if processed_data['BIRTH_DATE'].notna().sum() == 0:
            st.warning("All BIRTH_DATE values failed to convert. Trying more aggressive approach...")

            st.write("Sample BIRTH_DATE values before aggressive cleanup:")
            st.write(data['BIRTH_DATE'].dropna().head(5).tolist())

            def extract_date_components(date_str):
                if pd.isna(date_str):
                    return None
                date_str = str(date_str).strip()
                # Coba ekstrak dengan regex: MM/DD/YYYY atau DD/MM/YYYY
                match = re.search(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4}|\d{2})', date_str)
                if match:
                    first, second, third = match.groups()
                    first, second, third = int(first), int(second), int(third)
                    # Jika tahun hanya dua digit
                    if third < 100:
                        third = third + 1900 if third >= 50 else third + 2000

                    # Jika nilai kedua lebih besar dari 12, asumsikan format MM/DD/YYYY
                    if second > 12:
                        day, month = first, second
                    else:
                        # Jika ambigu, gunakan format default: first sebagai day
                        day, month = first, second

                    try:
                        return pd.Timestamp(year=third, month=month, day=day)
                    except Exception:
                        return None

                # Coba format YYYY-MM-DD
                match = re.search(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', date_str)
                if match:
                    year, month, day = map(int, match.groups())
                    try:
                        return pd.Timestamp(year=year, month=month, day=day)
                    except Exception:
                        return None
                return None

            processed_data['BIRTH_DATE'] = processed_data['BIRTH_DATE'].apply(extract_date_components)
            success_rate = processed_data['BIRTH_DATE'].notna().sum() / len(processed_data)
            st.write(f"Extracted BIRTH_DATE with component extraction: {processed_data['BIRTH_DATE'].notna().sum()} valid dates ({success_rate:.1%})")

        # Jika sudah ada tanggal yang valid, hitung usia
        if processed_data['BIRTH_DATE'].notna().sum() > 0:
            st.write("Calculating age from BIRTH_DATE...")
            current_date = pd.Timestamp.now()
            processed_data['Usia'] = ((current_date - processed_data['BIRTH_DATE']).dt.days / 365.25).round()

            # Filter usia yang valid (antara 18 dan 100 tahun)
            valid_age = (processed_data['Usia'] >= 18) & (processed_data['Usia'] <= 100)
            
            # Update mask untuk menyimpan baris dengan usia valid
            valid_data_mask = valid_data_mask & (valid_age | processed_data['BIRTH_DATE'].isna())
            
            # Set usia yang tidak valid menjadi NaN (akan dihapus di langkah berikutnya)
            processed_data.loc[~valid_age, 'Usia'] = np.nan

            processed_data['Usia'] = pd.to_numeric(processed_data['Usia'], errors='coerce')

            if processed_data['Usia'].notna().sum() > 0:
                st.success(f"Successfully calculated age for {processed_data['Usia'].notna().sum()} customers")
                st.write(f"Age statistics: Min={processed_data['Usia'].min():.1f}, Max={processed_data['Usia'].max():.1f}, Mean={processed_data['Usia'].mean():.1f}")

                # Buat kategori usia
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
                processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype(str)
                
                # IMPORTANT: Remove rows with 'nan' in age category
                nan_age_cat = processed_data['Usia_Kategori'].str.contains('nan', case=False, na=True)
                valid_data_mask = valid_data_mask & (~nan_age_cat)

                st.write("Age category distribution (before filtering):")
                st.write(processed_data['Usia_Kategori'].value_counts())
            else:
                st.warning("No valid ages could be calculated (between 18-100 years)")
                # Update mask untuk menandai semua baris sebagai tidak valid jika Usia diperlukan
                if 'Usia' in processed_data.columns:
                    valid_data_mask = valid_data_mask & False
        else:
            st.warning("No valid BIRTH_DATE values could be converted")
            # Update mask untuk menandai semua baris sebagai tidak valid jika BIRTH_DATE diperlukan
            if 'BIRTH_DATE' in processed_data.columns:
                valid_data_mask = valid_data_mask & False
    else:
        st.warning("BIRTH_DATE column not found in the data")

    # Konversi kolom numerik
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                    'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'LAST_MPF_TOP', 'AVG_MPF_INST',
                    'PRINCIPAL', 'GRS_DP', 'JMH_CON_SBLM_MPF', 'JMH_PPC']
    for col in numeric_cols:
        if col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].astype(str)\
                                            .str.replace(',', '')\
                                            .str.replace(r'[^\d\.-]', '', regex=True)
            
            # Konversi ke numerik
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            
            # Update mask untuk kolom numerik penting
            if col in ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF']:
                valid_data_mask = valid_data_mask & processed_data[col].notna()

    # Filter data untuk menghapus baris dengan NaN pada kolom penting
    filtered_data = processed_data[valid_data_mask].copy()
    
    # Tambahkan fitur tambahan
    filtered_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "TOTAL_PRODUCT_MPF" in filtered_data.columns:
        filtered_data["Multi-Transaction_Customer"] = filtered_data["TOTAL_PRODUCT_MPF"].astype(float).apply(lambda x: 1 if x > 1 else 0)

    # ADDITIONAL CHECK: Hapus semua "Unknown" nilai secara eksplisit
    for col in filtered_data.columns:
        if filtered_data[col].dtype == 'object' or col.endswith('_Kategori'):
            # Check for and remove rows with 'Unknown', 'nan', or empty values
            unknown_mask = filtered_data[col].str.contains('Unknown', case=False, na=True) | \
                         filtered_data[col].str.contains('nan', case=False, na=True) | \
                         filtered_data[col].str.contains('NaN', case=False, na=True) | \
                         filtered_data[col].str.contains('None', case=False, na=True) | \
                         filtered_data[col].isna() | \
                         (filtered_data[col].astype(str).str.strip() == '')
                         
            if unknown_mask.any():
                st.warning(f"Found {unknown_mask.sum()} rows with unknown/NaN values in {col}. These will be removed.")
                filtered_data = filtered_data[~unknown_mask]
    
    # Ensure all numeric columns have valid values
    for col in filtered_data.select_dtypes(include=['int64', 'float64']).columns:
        if filtered_data[col].isna().any():
            nan_mask = filtered_data[col].isna()
            st.warning(f"Found {nan_mask.sum()} rows with NaN in numeric column {col}. These will be removed.")
            filtered_data = filtered_data[~nan_mask]

    # Final check to ensure no NaN values remain in ANY column
    for col in filtered_data.columns:
        if filtered_data[col].isna().any():
            nan_mask = filtered_data[col].isna()
            st.warning(f"Found {nan_mask.sum()} rows with NaN in column {col} after primary cleaning. These will be removed.")
            filtered_data = filtered_data[~nan_mask]

    # Tampilkan informasi tentang data yang difilter
    original_rows = len(processed_data)
    filtered_rows = len(filtered_data)
    rows_removed = original_rows - filtered_rows
    
    st.info(f"Data validation: {filtered_rows} valid rows out of {original_rows} total rows. {rows_removed} rows with invalid/missing values were removed.")
    
    if filtered_rows == 0:
        st.error("No valid data rows remaining after filtering. Please check your data quality or adjust validation criteria.")
        return None
    
    return filtered_data

# Function to create example data (clean by default)
def create_example_data():
    """
    Create a sample customer dataset for testing and demonstration purposes.
    
    Returns:
    --------
    pandas.DataFrame
        Example dataset with essential columns
    """
    import pandas as pd

    data = pd.DataFrame({
        'CUST_NO': ['C001', 'C002', 'C003', 'C004'],
        'BIRTH_DATE': ['1990-01-01', '1985-05-15', '2000-10-30', '1975-07-20'],
        'CUST_SEX': ['F', 'M', 'F', 'M'],
        'TOTAL_AMOUNT_MPF': [5000000, 3000000, 7000000, 2000000],
        'TOTAL_PRODUCT_MPF': [2, 1, 3, 1],
        'LAST_MPF_DATE': ['2023-12-01', '2023-10-20', '2024-02-15', '2023-08-10'],
        'MAX_MPF_AMOUNT': [3000000, 2000000, 4000000, 1500000],
        'MIN_MPF_AMOUNT': [1000000, 1000000, 1500000, 500000],
        'LAST_MPF_AMOUNT': [2000000, 1000000, 3000000, 1000000],
        'LAST_MPF_INST': [12, 6, 18, 9],
        'LAST_MPF_TOP': [24, 12, 36, 18],
        'AVG_MPF_INST': [10, 5, 15, 8],
        'PRINCIPAL': [3000000, 2000000, 5000000, 1000000],
        'GRS_DP': [500000, 400000, 600000, 200000],
        'JMH_CON_SBLM_MPF': [1, 0, 2, 0],
        'JMH_PPC': [3, 1, 5, 0],
        'MPF_CATEGORIES_TAKEN': ['Motor', 'Elektronik', 'Motor', 'Furnitur'],
        'EDU_TYPE': ['S1', 'SMA', 'D3', 'S2'],
        'MARITAL_STAT': ['M', 'S', 'S', 'M']
    })

    # Convert date columns to datetime
    for date_col in ['BIRTH_DATE', 'LAST_MPF_DATE']:
        data[date_col] = pd.to_datetime(data[date_col])

    # Calculate age
    current_date = pd.Timestamp.now()
    data['Usia'] = ((current_date - data['BIRTH_DATE']).dt.days / 365.25).round()
    
    # Create age categories
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['<25', '25-35', '35-45', '45-55', '55+']
    data['Usia_Kategori'] = pd.cut(data['Usia'], bins=bins, labels=labels, right=False)
    
    # Convert category to string to ensure consistent processing
    data['Usia_Kategori'] = data['Usia_Kategori'].astype(str)
    
    # Add Multi-Transaction flag
    data['Multi-Transaction_Customer'] = data['TOTAL_PRODUCT_MPF'].apply(lambda x: 1 if x > 1 else 0)
    
    return data
