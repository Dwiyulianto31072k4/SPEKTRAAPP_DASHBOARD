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
            
            # Deteksi otomatis kolom numerik dan kategorikal
            potential_numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                              'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'MONTH_INST', 'OCPT_CODE', 'NO_OF_DEPEND']
            numeric_cols = [col for col in potential_numeric_cols if col in data.columns]
            
            potential_cat_cols = ['MPF_CATEGORIES_TAKEN', 'CUST_SEX', 'EDU_TYPE', 'HOUSE_STAT', 'MARITAL_STAT', 'AREA']
            cat_cols = [col for col in potential_cat_cols if col in data.columns]
            
            st.markdown(f"🔢 Auto-detected numeric columns: `{', '.join(numeric_cols)}`")
            st.markdown(f"📊 Auto-detected categorical columns: `{', '.join(cat_cols)}`")

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
            
            # Atur validasi untuk kolom tanggal
            date_format_options = st.multiselect(
                "Expected date formats to try (in order of priority):",
                options=['%Y%m%d', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y'],
                default=['%Y%m%d', '%Y-%m-%d']
            )
            
            # Pengaturan untuk outlier
            handle_outliers = st.checkbox("Detect and handle outliers in numeric columns", value=False)
            outlier_threshold = st.slider(
                "Z-score threshold for outliers:",
                min_value=2.0, max_value=5.0, value=3.0, step=0.1,
                help="Values with Z-score above this threshold will be considered outliers"
            )
            
            # Slider untuk menentukan minimum persentase baris valid yang dipertahankan
            min_valid_rows = st.slider(
                "Minimum percentage of valid rows required:",
                min_value=1, max_value=100, value=50,
                help="If the percentage of valid rows falls below this threshold, preprocessing will fail."
            )
            
            # ALWAYS enable strict validation
            strict_validation = True
            st.info("✅ Strict validation is enabled: All unknown/null values will be removed to ensure clean data.")

            # Tambahkan opsi visualisasi data quality
            show_data_quality_viz = st.checkbox("Show data quality visualization after preprocessing", value=True)

            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    # Panggil fungsi preprocess_data dengan pengaturan yang ditingkatkan
                    processed_data, preprocessing_logs = preprocess_data(
                        data, 
                        date_cols=date_cols,
                        numeric_cols=numeric_cols,
                        cat_cols=cat_cols,
                        required_cols=required_cols,
                        date_formats=date_format_options,
                        handle_outliers=handle_outliers,
                        outlier_threshold=outlier_threshold
                    )
                    
                    # Tampilkan log preprocessing
                    with st.expander("Preprocessing Log", expanded=False):
                        for log in preprocessing_logs:
                            st.text(log)
                    
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
                    
                    # Tambahkan fitur bisnis tambahan
                    final_data = calculate_business_features(final_data)
                    
                    # Tambahkan visualisasi kualitas data jika diminta
                    if show_data_quality_viz:
                        st.markdown("### Data Quality Visualization")
                        show_data_quality_charts(final_data)
                    
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
                            age_counts = final_data['Usia_Kategori'].value_counts()
                            
                            # Create bar chart for age categories
                            import plotly.express as px
                            fig = px.bar(
                                x=age_counts.index,
                                y=age_counts.values,
                                title="Age Category Distribution",
                                labels={"x": "Age Category", "y": "Count"},
                                color=age_counts.values,
                                color_continuous_scale=px.colors.sequential.Blues
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
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
            
            # Tambahkan fitur bisnis
            example_data = calculate_business_features(example_data)
            
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

def preprocess_data(data, date_cols, numeric_cols, cat_cols, required_cols=None, 
                  date_formats=None, handle_outliers=False, outlier_threshold=3.0):
    """
    Fungsi yang ditingkatkan untuk memproses data sebelum analisis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data mentah yang akan diproses
    date_cols : list
        Daftar kolom yang berisi tanggal
    numeric_cols : list
        Daftar kolom numerik
    cat_cols : list
        Daftar kolom kategorikal
    required_cols : list, optional
        Daftar kolom yang wajib memiliki nilai valid
    date_formats : list, optional
        Daftar format tanggal yang akan dicoba
    handle_outliers : bool, default=False
        Apakah perlu menangani outlier
    outlier_threshold : float, default=3.0
        Threshold Z-score untuk mendeteksi outlier
    
    Returns:
    --------
    tuple
        (pandas.DataFrame, list) - Data yang telah diproses dan log preprocessing
    """
    import pandas as pd
    import numpy as np
    import datetime
    import re
    
    processed_data = data.copy()
    preprocessing_log = []  # Simpan log untuk ditampilkan ke pengguna
    
    preprocessing_log.append(f"Starting preprocessing for {len(processed_data)} rows and {len(processed_data.columns)} columns")
    
    valid_data_mask = np.ones(len(processed_data), dtype=bool)  # Mask untuk menyimpan baris valid
    
    # Gunakan format tanggal default jika tidak disediakan
    if date_formats is None:
        date_formats = ['%Y%m%d', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y']
    
    # 1. Konversi kolom tanggal dengan pendekatan yang lebih fleksibel
    for col in date_cols:
        if col in processed_data.columns:
            preprocessing_log.append(f"Converting date column: {col}")
            
            # Cek apakah kolom tersebut merupakan tipe numerik (mis. Excel serial number)
            if pd.api.types.is_numeric_dtype(processed_data[col]) or processed_data[col].str.isnumeric().all():
                try:
                    processed_data[col] = pd.to_datetime(processed_data[col], origin='1899-12-30', unit='d', errors='coerce')
                    valid_count = processed_data[col].notna().sum()
                    preprocessing_log.append(f"  Converted {col} from numeric (Excel serial) to datetime: {valid_count} valid dates")
                except Exception as e:
                    preprocessing_log.append(f"  Numeric conversion failed for {col} with error: {e}")

            # Jika bukan numeric, lakukan clean-up terlebih dahulu
            original_values = processed_data[col].copy()
            processed_data[col] = processed_data[col].apply(clean_date_string)

            # Coba semua format tanggal yang ditentukan
            converted = False

            for fmt in date_formats:
                try:
                    temp_dates = pd.to_datetime(processed_data[col], format=fmt, errors='coerce')
                    success_rate = temp_dates.notna().sum() / len(processed_data)
                    
                    if success_rate > 0.3:  # Jika >30% berhasil dikonversi
                        processed_data[col] = temp_dates
                        preprocessing_log.append(f"  Converted {col} using format {fmt}: {temp_dates.notna().sum()} valid dates ({success_rate:.1%})")
                        converted = True
                        break
                except Exception:
                    continue

            # Fallback ke pandas automatic detection jika format spesifik gagal
            if not converted:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                success_rate = processed_data[col].notna().sum() / len(processed_data)
                preprocessing_log.append(f"  Converted {col} using pandas automatic detection: {processed_data[col].notna().sum()} valid dates ({success_rate:.1%})")
            
            # Update mask untuk validasi tanggal jika kolom ini diperlukan
            if required_cols and col in required_cols:
                valid_data_mask = valid_data_mask & processed_data[col].notna()

    # 2. Penanganan khusus untuk BIRTH_DATE dan perhitungan Usia
    if 'BIRTH_DATE' in processed_data.columns:
        preprocessing_log.append("Processing BIRTH_DATE for age calculation")
        valid_birth_dates = processed_data['BIRTH_DATE'].notna().sum()
        
        if valid_birth_dates == 0:
            preprocessing_log.append("  All BIRTH_DATE values failed to convert. Trying more aggressive approach...")

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
            valid_birth_dates = processed_data['BIRTH_DATE'].notna().sum()
            preprocessing_log.append(f"  Extracted BIRTH_DATE with component extraction: {valid_birth_dates} valid dates ({valid_birth_dates/len(processed_data):.1%})")

        # Jika sudah ada tanggal yang valid, hitung usia
        if valid_birth_dates > 0:
            preprocessing_log.append("Calculating age from BIRTH_DATE...")
            reference_date = pd.Timestamp.now()
            
            # Pendekatan yang lebih akurat untuk perhitungan usia (menggunakan jumlah hari)
            processed_data['Usia'] = ((reference_date - processed_data['BIRTH_DATE']).dt.days / 365.25).round()

            # Filter usia yang valid (antara 18 dan 100 tahun)
            valid_age = (processed_data['Usia'] >= 18) & (processed_data['Usia'] <= 100)
            invalid_age_count = (~valid_age & processed_data['Usia'].notna()).sum()
            
            if invalid_age_count > 0:
                preprocessing_log.append(f"  Found {invalid_age_count} customers with age outside valid range (18-100 years)")
                
                # Update mask untuk menyimpan baris dengan usia valid jika diperlukan
                if required_cols and 'BIRTH_DATE' in required_cols:
                    valid_data_mask = valid_data_mask & (valid_age | processed_data['BIRTH_DATE'].isna())
                
                # Set usia yang tidak valid menjadi NaN
                processed_data.loc[~valid_age & processed_data['Usia'].notna(), 'Usia'] = np.nan

            processed_data['Usia'] = pd.to_numeric(processed_data['Usia'], errors='coerce')

            valid_age_count = processed_data['Usia'].notna().sum()
            if valid_age_count > 0:
                preprocessing_log.append(f"  Successfully calculated age for {valid_age_count} customers")
                stats = processed_data['Usia'].describe()
                preprocessing_log.append(f"  Age statistics: Min={stats['min']:.1f}, Max={stats['max']:.1f}, Mean={stats['mean']:.1f}")

                # Buat kategori usia
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
                processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype(str)
                
                # IMPORTANT: Remove rows with 'nan' in age category
                nan_age_cat = processed_data['Usia_Kategori'].str.contains('nan', case=False, na=True)
                if nan_age_cat.any():
                    valid_data_mask = valid_data_mask & (~nan_age_cat)
                    preprocessing_log.append(f"  Removed {nan_age_cat.sum()} rows with invalid age categories")

                age_distribution = processed_data['Usia_Kategori'].value_counts().to_dict()
                preprocessing_log.append(f"  Age category distribution: {age_distribution}")
            else:
                preprocessing_log.append("  No valid ages could be calculated (between 18-100 years)")
                # Update mask untuk menandai baris sebagai tidak valid jika Usia diperlukan
                if required_cols and 'Usia' in required_cols:
                    valid_data_mask = valid_data_mask & False
        else:
            preprocessing_log.append("No valid BIRTH_DATE values could be converted")
            # Update mask untuk menandai baris sebagai tidak valid jika BIRTH_DATE diperlukan
            if required_cols and 'BIRTH_DATE' in required_cols:
                valid_data_mask = valid_data_mask & False
    else:
        preprocessing_log.append("BIRTH_DATE column not found in the data")

    # 3. Konversi kolom numerik dengan penanganan lebih baik
    for col in numeric_cols:
        if col in processed_data.columns:
            preprocessing_log.append(f"Converting numeric column: {col}")
            
            # Jika kolom adalah string, hapus karakter non-numerik
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].astype(str)\
                                            .str.replace(',', '')\
                                            .str.replace(r'[^\d\.-]', '', regex=True)
            
            # Konversi ke numerik
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            valid_numeric = processed_data[col].notna().sum()
            preprocessing_log.append(f"  Converted {col} to numeric: {valid_numeric} valid values ({valid_numeric/len(processed_data):.1%})")
            
            # Deteksi dan tangani outlier jika diminta
            if handle_outliers and processed_data[col].notna().sum() > 0:
                z_scores = np.abs((processed_data[col] - processed_data[col].mean()) / processed_data[col].std())
                outliers = z_scores > outlier_threshold
                
                if outliers.any():
                    outlier_count = outliers.sum()
                    preprocessing_log.append(f"  Detected {outlier_count} outliers in {col} (Z-score > {outlier_threshold})")
                    
                    # Replace outliers with median
                    median_val = processed_data[col].median()
                    processed_data.loc[outliers, col] = median_val
                    preprocessing_log.append(f"  Replaced outliers with median value: {median_val}")
            
            # Update mask untuk kolom numerik penting
            if required_cols and col in required_cols:
                valid_data_mask = valid_data_mask & processed_data[col].notna()

    # 4. Bersihkan kolom kategorikal
    for col in cat_cols:
        if col in processed_data.columns:
            preprocessing_log.append(f"Cleaning categorical column: {col}")
            
           # Hapus whitespace
           if processed_data[col].dtype == 'object':
               processed_data[col] = processed_data[col].str.strip()
               preprocessing_log.append(f"  Removed whitespace from {col}")
           
           # Standardisasi nilai
           if col == 'CUST_SEX':
               # Standardisasi jenis kelamin
               processed_data[col] = processed_data[col].replace({'MALE': 'M', 'FEMALE': 'F', 'L': 'M', 'P': 'F'})
               preprocessing_log.append(f"  Standardized values in {col}: MALE/L -> M, FEMALE/P -> F")
           
           # Deteksi dan tangani nilai yang hilang atau tidak valid
           missing_or_invalid = processed_data[col].isna() | (processed_data[col] == '') | \
                               (processed_data[col].astype(str).str.lower() == 'nan') | \
                               (processed_data[col].astype(str).str.lower() == 'none') | \
                               (processed_data[col].astype(str).str.lower() == 'unknown')
           
           if missing_or_invalid.any():
               missing_count = missing_or_invalid.sum()
               preprocessing_log.append(f"  Found {missing_count} missing or invalid values in {col}")
               
               if required_cols and col in required_cols:
                   valid_data_mask = valid_data_mask & (~missing_or_invalid)
                   preprocessing_log.append(f"  Flagged rows with missing {col} values for removal")
               else:
                   mode_val = processed_data.loc[~missing_or_invalid, col].mode().iloc[0] if (~missing_or_invalid).any() else "Unknown"
                   processed_data.loc[missing_or_invalid, col] = mode_val
                   preprocessing_log.append(f"  Filled missing values with most common value: {mode_val}")

   # 5. Filter data untuk menghapus baris dengan NaN pada kolom penting
   valid_rows_before = valid_data_mask.sum()
   preprocessing_log.append(f"Initial valid rows: {valid_rows_before} out of {len(processed_data)}")
   
   filtered_data = processed_data[valid_data_mask].copy()
   preprocessing_log.append(f"After filtering: {len(filtered_data)} valid rows remain ({len(filtered_data)/len(processed_data)*100:.1f}%)")
   
   if len(filtered_data) == 0:
       preprocessing_log.append("ERROR: No valid rows remain after filtering!")
       return None, preprocessing_log

   # 6. Tambahkan fitur tambahan
   filtered_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   if "TOTAL_PRODUCT_MPF" in filtered_data.columns:
       filtered_data["Multi-Transaction_Customer"] = filtered_data["TOTAL_PRODUCT_MPF"].astype(float).apply(lambda x: 1 if x > 1 else 0)
       multi_prod_pct = filtered_data["Multi-Transaction_Customer"].mean() * 100
       preprocessing_log.append(f"Added Multi-Transaction_Customer flag: {multi_prod_pct:.1f}% customers have multiple products")

   # 7. Periksa untuk memastikan tidak ada nilai yang hilang di kolom penting
   critical_missing = {}
   for col in filtered_data.columns:
       if filtered_data[col].isna().any():
           missing_count = filtered_data[col].isna().sum()
           critical_missing[col] = missing_count
           
   if critical_missing:
       preprocessing_log.append("WARNING: Still found missing values in the following columns:")
       for col, count in critical_missing.items():
           preprocessing_log.append(f"  - {col}: {count} missing values")
   else:
       preprocessing_log.append("✅ No missing values in the filtered data!")

   return filtered_data, preprocessing_log

def calculate_business_features(data):
   """
   Menghitung fitur bisnis tambahan yang berguna untuk analisis
   
   Parameters:
   -----------
   data : pandas.DataFrame
       Data yang sudah dibersihkan
   
   Returns:
   --------
   pandas.DataFrame
       Data dengan fitur bisnis tambahan
   """
   result_data = data.copy()
   
   # 1. Recency (berapa hari sejak transaksi terakhir)
   if 'LAST_MPF_DATE' in result_data.columns and result_data['LAST_MPF_DATE'].notna().any():
       reference_date = pd.Timestamp.now()
       result_data['Recency_Days'] = (reference_date - result_data['LAST_MPF_DATE']).dt.days
       
       # Buat kategori recency
       result_data['Recency_Category'] = pd.cut(
           result_data['Recency_Days'],
           bins=[0, 30, 90, 180, 365, float('inf')],
           labels=['Very Recent', 'Recent', 'Moderate', 'Lapsed', 'Inactive'],
           include_lowest=True
       )
   
   # 2. Tenure (berapa lama menjadi pelanggan)
   if 'FIRST_MPF_DATE' in result_data.columns and 'LAST_MPF_DATE' in result_data.columns:
       valid_mask = result_data['FIRST_MPF_DATE'].notna() & result_data['LAST_MPF_DATE'].notna()
       result_data['Customer_Tenure_Days'] = np.nan
       
       if valid_mask.any():
           result_data.loc[valid_mask, 'Customer_Tenure_Days'] = (
               result_data.loc[valid_mask, 'LAST_MPF_DATE'] - 
               result_data.loc[valid_mask, 'FIRST_MPF_DATE']
           ).dt.days
           
           # Convert to years for easier interpretation
           result_data['Customer_Tenure_Years'] = result_data['Customer_Tenure_Days'] / 365.25
           
           # Create tenure categories
           result_data['Tenure_Category'] = pd.cut(
               result_data['Customer_Tenure_Years'],
               bins=[0, 1, 3, 5, 10, float('inf')],
               labels=['New', 'Developing', 'Established', 'Loyal', 'Long-term'],
               include_lowest=True
           )
   
   # 3. Customer Value (dari monetary value)
   if 'TOTAL_AMOUNT_MPF' in result_data.columns and result_data['TOTAL_AMOUNT_MPF'].notna().any():
       quantiles = result_data['TOTAL_AMOUNT_MPF'].quantile([0.2, 0.4, 0.6, 0.8])
       result_data['Value_Category'] = pd.cut(
           result_data['TOTAL_AMOUNT_MPF'],
           bins=[0, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], float('inf')],
           labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
           include_lowest=True
       )
   
   # 4. Multi-Product Customer Flag
   if 'TOTAL_PRODUCT_MPF' in result_data.columns:
       result_data['Multi_Product_Flag'] = (result_data['TOTAL_PRODUCT_MPF'] > 1).astype(int)
   
   # 5. Customer Segment Combination (Value + Recency)
   if 'Value_Category' in result_data.columns and 'Recency_Category' in result_data.columns:
       # Ensure they're string type
       result_data['Value_Category'] = result_data['Value_Category'].astype(str)
       result_data['Recency_Category'] = result_data['Recency_Category'].astype(str)
       
       # Create combined segment
       result_data['Customer_Segment'] = result_data['Value_Category'] + ' / ' + result_data['Recency_Category']
       
       # Simplify to high-level segments
       conditions = [
           (result_data['Value_Category'].isin(['High', 'Medium-High'])) & 
           (result_data['Recency_Category'].isin(['Very Recent', 'Recent'])),
           
           (result_data['Value_Category'].isin(['High', 'Medium-High'])) & 
           (result_data['Recency_Category'].isin(['Moderate', 'Lapsed', 'Inactive'])),
           
           (result_data['Value_Category'].isin(['Medium'])) & 
           (result_data['Recency_Category'].isin(['Very Recent', 'Recent'])),
           
           (result_data['Value_Category'].isin(['Medium'])) & 
           (result_data['Recency_Category'].isin(['Moderate', 'Lapsed', 'Inactive'])),
           
           (result_data['Value_Category'].isin(['Low', 'Medium-Low'])) & 
           (result_data['Recency_Category'].isin(['Very Recent', 'Recent'])),
           
           (result_data['Value_Category'].isin(['Low', 'Medium-Low'])) & 
           (result_data['Recency_Category'].isin(['Moderate', 'Lapsed', 'Inactive']))
       ]
       
       choices = [
           'High Value Active',
           'High Value Inactive',
           'Medium Value Active',
           'Medium Value Inactive',
           'Low Value Active',
           'Low Value Inactive'
       ]
       
       result_data['Customer_Segment_Simple'] = np.select(conditions, choices, default='Unknown')
   
   return result_data

def show_data_quality_charts(data):
   """
   Menampilkan visualisasi kualitas data
   
   Parameters:
   -----------
   data : pandas.DataFrame
       Data yang sudah dibersihkan
   """
   import plotly.express as px
   import plotly.graph_objects as go
   
   # 1. Data Completeness
   st.subheader("Data Completeness")
   
   all_cols = data.columns
   col_completeness = [(col, data[col].notna().mean() * 100) for col in all_cols]
   col_completeness.sort(key=lambda x: x[1])
   
   col_names = [item[0] for item in col_completeness]
   completeness_pct = [item[1] for item in col_completeness]
   
   fig = px.bar(
       x=completeness_pct,
       y=col_names,
       orientation='h',
       title="Data Completeness by Column (%)",
       labels={"x": "Completeness (%)", "y": "Column"},
       color=completeness_pct,
       color_continuous_scale=px.colors.sequential.Blues
   )
   
   fig.add_shape(
       type="line",
       x0=100, y0=-0.5,
       x1=100, y1=len(col_names)-0.5,
       line=dict(color="green", width=3, dash="dash")
   )
   
   fig.update_layout(height=max(400, len(col_names) * 20))
   st.plotly_chart(fig, use_container_width=True)
   
   # 2. Data Types Distribution
   st.subheader("Data Types Distribution")
   
   dtype_counts = {}
   for dtype in data.dtypes.unique():
       dtype_counts[str(dtype)] = (data.dtypes == dtype).sum()
   
   fig = px.pie(
       values=list(dtype_counts.values()),
       names=list(dtype_counts.keys()),
       title="Distribution of Column Data Types",
       color_discrete_sequence=px.colors.qualitative.Set3
   )
   
   st.plotly_chart(fig, use_container_width=True)
   
   # 3. Categorical Values Distribution (for selected columns)
   st.subheader("Categorical Values Distribution")
   
   cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
   if cat_cols:
       # Let user select which categorical column to view
       selected_cat_col = st.selectbox("Select categorical column:", cat_cols)
       
       value_counts = data[selected_cat_col].value_counts().reset_index()
       value_counts.columns = [selected_cat_col, 'Count']
       
       if len(value_counts) > 10:
           value_counts = pd.concat([
               value_counts.head(9),
               pd.DataFrame({
                   selected_cat_col: ['Others'],
                   'Count': [value_counts.iloc[9:]['Count'].sum()]
               })
           ]).reset_index(drop=True)
       
       fig = px.pie(
           value_counts,
           values='Count',
           names=selected_cat_col,
           title=f"Distribution of {selected_cat_col} Values",
           color_discrete_sequence=px.colors.qualitative.Pastel
       )
       
       st.plotly_chart(fig, use_container_width=True)
   else:
       st.info("No categorical columns found in the data.")

def create_example_data():
   """
   Create a sample customer dataset for testing and demonstration purposes.
   
   Returns:
   --------
   pandas.DataFrame
       Example dataset with essential columns
   """
   import pandas as pd
   import numpy as np
   from datetime import datetime, timedelta

   # Create a base dataset with 100 customers
   np.random.seed(42)  # For reproducibility
   n_customers = 100
   
   # Customer IDs
   cust_ids = [f'C{i:04d}' for i in range(1, n_customers + 1)]
   
   # Dates (ensure they follow a realistic pattern)
   today = datetime.now()
   
   birth_dates = [today - timedelta(days=np.random.randint(18*365, 70*365)) for _ in range(n_customers)]
   
   first_ppc_dates = [today - timedelta(days=np.random.randint(30, 1825)) for _ in range(n_customers)]
   
   first_mpf_dates = [first_ppc + timedelta(days=np.random.randint(30, 365)) 
                     for first_ppc in first_ppc_dates]
   
   last_mpf_dates = [first_mpf + timedelta(days=np.random.randint(30, 730)) 
                    for first_mpf in first_mpf_dates]
   
   contract_active_dates = [first_mpf + timedelta(days=np.random.randint(1, 60)) 
                           for first_mpf in first_mpf_dates]
   
   # Numeric values
   total_amounts = np.random.lognormal(mean=16, sigma=1, size=n_customers).astype(int)  # Realistic money amounts
   total_products = np.random.choice([1, 2, 3, 4, 5], size=n_customers, p=[0.4, 0.3, 0.2, 0.08, 0.02])
   month_insts = np.random.choice([6, 12, 18, 24, 36], size=n_customers)
   
   # Categorical values
   genders = np.random.choice(['M', 'F'], size=n_customers, p=[0.6, 0.4])
   product_categories = np.random.choice(
       ['Motor', 'Elektronik', 'Furnitur', 'Gadget', 'Lainnya'],
       size=n_customers,
       p=[0.4, 0.25, 0.15, 0.15, 0.05]
   )
   education_types = np.random.choice(
       ['SD', 'SMP', 'SMA', 'D3', 'S1', 'S2'],
       size=n_customers,
       p=[0.05, 0.1, 0.4, 0.15, 0.25, 0.05]
   )
   house_stats = np.random.choice(['Milik Sendiri', 'Sewa', 'Keluarga'], size=n_customers, p=[0.5, 0.3, 0.2])
   marital_stats = np.random.choice(['M', 'S', 'D'], size=n_customers, p=[0.6, 0.35, 0.05])
   areas = np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar'], size=n_customers)
   
   # Occupation and dependents
   ocpt_codes = np.random.randint(1, 10, size=n_customers)
   no_of_depends = np.random.choice([0, 1, 2, 3, 4, 5], size=n_customers, p=[0.2, 0.15, 0.3, 0.2, 0.1, 0.05])
   
   # Create the DataFrame
   data = pd.DataFrame({
       'CUST_NO': cust_ids,
       'BIRTH_DATE': birth_dates,
       'CUST_SEX': genders,
       'TOTAL_AMOUNT_MPF': total_amounts,
       'TOTAL_PRODUCT_MPF': total_products,
       'FIRST_PPC_DATE': first_ppc_dates,
       'FIRST_MPF_DATE': first_mpf_dates,
       'LAST_MPF_DATE': last_mpf_dates,
       'CONTRACT_ACTIVE_DATE': contract_active_dates,
       'MONTH_INST': month_insts,
       'MPF_CATEGORIES_TAKEN': product_categories,
       'EDU_TYPE': education_types,
       'HOUSE_STAT': house_stats,
       'MARITAL_STAT': marital_stats,
       'AREA': areas,
       'OCPT_CODE': ocpt_codes,
       'NO_OF_DEPEND': no_of_depends
   })
   
   # Add some calculated fields
   current_date = pd.Timestamp.now()
   data['Usia'] = ((current_date - pd.to_datetime(data['BIRTH_DATE'])).dt.days / 365.25).round().astype(int)
   
   # Create age categories
   bins = [0, 25, 35, 45, 55, 100]
   labels = ['<25', '25-35', '35-45', '45-55', '55+']
   data['Usia_Kategori'] = pd.cut(data['Usia'], bins=bins, labels=labels, right=False)
   
   # Add Multi-Transaction flag
   data['Multi-Transaction_Customer'] = (data['TOTAL_PRODUCT_MPF'] > 1).astype(int)
   
   # Add business days since last transaction
   data['Recency_Days'] = (current_date - pd.to_datetime(data['LAST_MPF_DATE'])).dt.days
   
   # Add Recency categories
   data['Recency_Category'] = pd.cut(
       data['Recency_Days'],
       bins=[0, 30, 90, 180, 365, float('inf')],
       labels=['Very Recent', 'Recent', 'Moderate', 'Lapsed', 'Inactive'],
       include_lowest=True
   )
   
   return data
            
