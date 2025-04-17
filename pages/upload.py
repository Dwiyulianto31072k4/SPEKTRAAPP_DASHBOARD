import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re

# Remove the decorator since its implementation isn't available
# and we'll implement strict validation directly
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
            st.success(f"File '{uploaded_file.name}' successfully loaded with {data.shape[0]} rows and {data.shape[1]} columns!")
            st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(data.head())

            # Deteksi otomatis kolom dengan 'DATE' di namanya
            date_cols = [col for col in data.columns if 'DATE' in col.upper()]
            st.markdown(f"🔍 Auto-detected date columns: `{', '.join(date_cols)}`")

            # Tambahkan pengaturan untuk validasi data
            st.markdown("### Data Validation Settings")
            
            required_cols = st.multiselect(
                "Select required columns (rows with NaN in these columns will be removed):",
                options=data.columns.tolist(),
                default=['CUST_NO', 'LAST_MPF_DATE', 'TOTAL_PRODUCT_MPF', 'TOTAL_AMOUNT_MPF'] if all(col in data.columns for col in ['CUST_NO', 'LAST_MPF_DATE', 'TOTAL_PRODUCT_MPF', 'TOTAL_AMOUNT_MPF']) else []
            )
            
            require_valid_age = st.checkbox("Require valid age data (18-100 years old)", value=True)
            
            min_valid_rows = st.slider(
                "Minimum percentage of valid rows required:",
                min_value=1, max_value=100, value=50,
                help="If the percentage of valid rows falls below this threshold, preprocessing will fail."
            )
            
            # Add new option to remove all unknown/null values
            strict_validation = st.checkbox("Enable strict validation (remove ALL unknown/null values)", value=True, 
                                           help="When enabled, any row with unknown or missing values in any column will be removed.")

            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    # Panggil fungsi preprocess_data yang telah dimodifikasi
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
                    
                    # NEW: Apply strict validation if enabled
                    if strict_validation:
                        for col in processed_data.columns:
                            if col != 'CUST_NO':  # Always keep customer ID column
                                # Remove rows with NaN
                                valid_rows_mask = valid_rows_mask & processed_data[col].notna()
                                
                                # Remove rows with 'Unknown' or 'unknown' values if column is categorical
                                if processed_data[col].dtype == 'object' or col.endswith('_Kategori'):
                                    unknown_mask = ~(processed_data[col].str.contains('Unknown', case=False, na=False) | 
                                                    processed_data[col].str.contains('nan', case=False, na=False))
                                    valid_rows_mask = valid_rows_mask & unknown_mask
                    
                    # Filter data berdasarkan mask
                    final_data = processed_data[valid_rows_mask].copy()
                    
                    # Periksa persentase baris valid
                    valid_percentage = (len(final_data) / len(data)) * 100
                    st.info(f"Valid data rows: {len(final_data)} out of {len(data)} ({valid_percentage:.1f}%)")
                    
                    if valid_percentage < min_valid_rows:
                        st.error(f"❌ Preprocessing failed: Only {valid_percentage:.1f}% valid rows (minimum required: {min_valid_rows}%)")
                        return
                    
                    # NEW: Final check to confirm no unknown values remain
                    unknown_values_found = False
                    unknown_columns = []
                    
                    for col in final_data.columns:
                        if final_data[col].dtype == 'object' or col.endswith('_Kategori'):
                            if any(final_data[col].str.contains('Unknown', case=False, na=False)) or final_data[col].isna().any():
                                unknown_values_found = True
                                unknown_columns.append(col)
                    
                    if unknown_values_found and strict_validation:
                        # Apply one more filter to catch any remaining unknowns
                        for col in unknown_columns:
                            mask = ~(final_data[col].str.contains('Unknown', case=False, na=False) | final_data[col].isna())
                            final_data = final_data[mask]
                        
                        st.warning(f"Removed additional rows with unknown values in columns: {', '.join(unknown_columns)}")
                    
                    # Tampilkan hasil preprocessing
                    st.success("✅ Data preprocessing completed!")
                    st.dataframe(final_data.head())

                    # Tampilkan informasi kolom Usia jika ada
                    if 'Usia' in final_data.columns:
                        st.markdown("### Age Information")
                        valid_ages = final_data['Usia'].notna().sum()
                        total_rows = len(final_data)
                        st.write(f"Valid age values: {valid_ages} out of {total_rows} rows ({valid_ages/total_rows*100:.1f}%)")

                        if valid_ages > 0:
                            age_stats = final_data['Usia'].describe()
                            st.write(f"Age statistics: Min={age_stats['min']:.1f}, Max={age_stats['max']:.1f}, Mean={age_stats['mean']:.1f}")

                            if 'Usia_Kategori' in final_data.columns:
                                st.write("Age category distribution:")
                                st.write(final_data['Usia_Kategori'].value_counts())
                    
                    # Final check confirmation
                    st.success("✅ All data is clean with no unknown or NaN values!")
                    
                    # Simpan data yang telah divalidasi ke session state
                    st.session_state.data = final_data

                    # Pastikan folder temp ada
                    if not os.path.exists("temp"):
                        os.makedirs("temp")

                    final_data.to_excel("temp/processed_data.xlsx", index=False)
                    st.session_state.eda_completed = True

                    st.markdown("### Next Steps")
                    st.info("You can now proceed to the Exploratory Data Analysis section.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.warning("Please check your file and try again.")
    else:
        st.write("Or use our example data to explore the application:")
        if st.button("Use Example Data"):
            example_data = create_example_data()
            
            # Validasi data contoh untuk memastikan tidak ada NaN
            example_data = example_data.dropna()
            
            # Pastikan ada kolom Usia dan kategori
            if 'BIRTH_DATE' in example_data.columns:
                current_date = pd.Timestamp.now()
                example_data['Usia'] = ((current_date - pd.to_datetime(example_data['BIRTH_DATE'])).dt.days / 365.25).round()
                
                # Buat kategori usia
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                example_data['Usia_Kategori'] = pd.cut(example_data['Usia'], bins=bins, labels=labels, right=False)
                
                # NEW: Ensure no unknown values in example data
                example_data = example_data.dropna()
            
            st.success("✅ Example data loaded successfully!")
            st.session_state.data = example_data
            st.session_state.uploaded_file_name = "example_data.xlsx"

            if not os.path.exists("temp"):
                os.makedirs("temp")
            example_data.to_excel("temp/processed_data.xlsx", index=False)
            st.dataframe(example_data.head())
            st.session_state.eda_completed = True
            st.markdown("### Next Steps")
            st.info("You can now proceed to the Exploratory Data Analysis section.")

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
                
                # MODIFIED: Convert to string and REMOVE 'nan' values completely, not convert to 'Unknown'
                processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype(str)
                processed_data.loc[processed_data['Usia_Kategori'] == 'nan', 'Usia_Kategori'] = np.nan
                
                # MODIFIED: Update mask to remove rows with missing age categories
                valid_age_cat = processed_data['Usia_Kategori'] != 'nan'
                valid_data_mask = valid_data_mask & valid_age_cat

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

    # MODIFIED: Final check to remove ANY row with 'Unknown' values in categorical columns
    for col in filtered_data.columns:
        if filtered_data[col].dtype == 'object' or col.endswith('_Kategori'):
            # Check for 'Unknown' or 'nan' strings
            unknown_mask = filtered_data[col].str.contains('Unknown', case=False, na=True) | \
                          filtered_data[col].str.contains('nan', case=False, na=True) | \
                          filtered_data[col].isna()
            
            if unknown_mask.any():
                st.warning(f"Found {unknown_mask.sum()} rows with unknown/NaN values in {col}. These will be removed.")
                filtered_data = filtered_data[~unknown_mask]

    # Tampilkan informasi tentang data yang difilter
    original_rows = len(processed_data)
    filtered_rows = len(filtered_data)
    rows_removed = original_rows - filtered_rows
    
    st.info(f"Data validation: {filtered_rows} valid rows out of {original_rows} total rows. {rows_removed} rows with invalid/missing values were removed.")
    
    if filtered_rows == 0:
        st.error("No valid data rows remaining after filtering. Please check your data quality or adjust validation criteria.")
        return None
    
    return filtered_data
