import pandas as pd
import numpy as np
import datetime

def preprocess_data(data, date_cols):
    """
    Enhanced preprocessing function that focuses on proper data validation
    and robust age calculation from customer data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw customer data uploaded by the user
    date_cols : list
        List of columns containing date information
    
    Returns:
    --------
    pandas.DataFrame
        Processed data ready for analysis
    """
    processed_data = data.copy()
    preprocessing_log = []
    
    # 1. Date column conversion with robust error handling
    for col in date_cols:
        try:
            # Try explicit format first (most reliable)
            if processed_data[col].dtype == 'object':
                # Check for common date formats
                sample = processed_data[col].dropna().iloc[0] if not processed_data[col].dropna().empty else ""
                
                # Log detected format for debugging
                preprocessing_log.append(f"Sample date in {col}: {sample}")
                
                if len(str(sample)) == 8 and str(sample).isdigit():
                    # Likely YYYYMMDD format
                    processed_data[col] = pd.to_datetime(processed_data[col], format='%Y%m%d', errors='coerce')
                    preprocessing_log.append(f"Converted {col} using YYYYMMDD format")
                elif '/' in str(sample):
                    # Try MM/DD/YYYY or DD/MM/YYYY
                    try:
                        processed_data[col] = pd.to_datetime(processed_data[col], format='%m/%d/%Y', errors='coerce')
                        preprocessing_log.append(f"Converted {col} using MM/DD/YYYY format")
                    except:
                        processed_data[col] = pd.to_datetime(processed_data[col], format='%d/%m/%Y', errors='coerce')
                        preprocessing_log.append(f"Converted {col} using DD/MM/YYYY format")
                elif '-' in str(sample):
                    # Try YYYY-MM-DD first (ISO format)
                    processed_data[col] = pd.to_datetime(processed_data[col], format='%Y-%m-%d', errors='coerce')
                    preprocessing_log.append(f"Converted {col} using YYYY-MM-DD format")
                else:
                    # Fall back to pandas' inference
                    processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                    preprocessing_log.append(f"Converted {col} using pandas' automatic format detection")
        except Exception as e:
            preprocessing_log.append(f"Error converting {col}: {str(e)}")
            # Still attempt conversion with pandas' inference as fallback
            processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
    
    # 2. Robust age calculation from BIRTH_DATE
    if 'BIRTH_DATE' in processed_data.columns:
        # Check if conversion was successful
        valid_dates = ~processed_data['BIRTH_DATE'].isnull()
        valid_count = valid_dates.sum()
        total_count = len(processed_data)
        
        preprocessing_log.append(f"Birth date conversion: {valid_count}/{total_count} valid dates ({valid_count/total_count*100:.1f}%)")
        
        if valid_count > 0:
            # Calculate reference date (today)
            reference_date = pd.Timestamp.now()
            
            # Create age column initialized with NaN
            processed_data['Usia'] = np.nan
            
            # Calculate ages properly (accounting for month/day, not just year)
            processed_data.loc[valid_dates, 'Usia'] = (
                (reference_date - processed_data.loc[valid_dates, 'BIRTH_DATE']).dt.days / 365.25
            ).astype(int)
            
            # Validate age range (between 18 and 100 for most financial applications)
            age_range_valid = (processed_data['Usia'] >= 18) & (processed_data['Usia'] <= 100)
            invalid_ages = (~age_range_valid) & (~processed_data['Usia'].isnull())
            
            if invalid_ages.any():
                preprocessing_log.append(f"Found {invalid_ages.sum()} customers with unusual ages. Setting to NaN.")
                processed_data.loc[invalid_ages, 'Usia'] = np.nan
            
            # Create meaningful age categories for segmentation
            # These categories should align with typical marketing segments
            age_bins = [18, 25, 35, 45, 55, 65, 100]
            age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            
            processed_data['Usia_Kategori'] = pd.cut(
                processed_data['Usia'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
            
            # For records with missing age, create an "Unknown" category
            processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype(str)
            processed_data.loc[processed_data['Usia'].isnull(), 'Usia_Kategori'] = 'Unknown'
            
            # Log age statistics
            preprocessing_log.append(f"Age statistics: Min={processed_data['Usia'].min():.1f}, Max={processed_data['Usia'].max():.1f}, Mean={processed_data['Usia'].mean():.1f}")
            preprocessing_log.append(f"Age category distribution: {processed_data['Usia_Kategori'].value_counts().to_dict()}")
        else:
            preprocessing_log.append("Warning: No valid birth dates found, cannot calculate customer ages")
            
            # Create placeholder age category for completeness
            processed_data['Usia'] = np.nan
            processed_data['Usia_Kategori'] = 'Unknown'
    
    # 3. Convert numeric columns to appropriate types
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                   'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'LAST_MPF_TOP', 'AVG_MPF_INST',
                   'PRINCIPAL', 'GRS_DP', 'JMH_CON_SBLM_MPF', 'JMH_PPC']
    
    for col in numeric_cols:
        if col in processed_data.columns:
            # Clean the data first (remove non-numeric characters)
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            
            # Convert to numeric, with warning for columns with high failure rate
            original_count = len(processed_data)
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            null_count = processed_data[col].isnull().sum()
            
            if null_count > 0.2 * original_count:  # If more than 20% conversion failed
                preprocessing_log.append(f"Warning: Column '{col}' has {null_count}/{original_count} null values after numeric conversion")
    
    # 4. Business-specific enhancements - identify high-value metrics
    if 'TOTAL_AMOUNT_MPF' in processed_data.columns and not processed_data['TOTAL_AMOUNT_MPF'].isnull().all():
        # Create customer value categories based on total transaction amount
        # These categories should be based on business knowledge of customer value tiers
        value_quantiles = processed_data['TOTAL_AMOUNT_MPF'].quantile([0.25, 0.5, 0.75, 0.9])
        
        processed_data['Value_Category'] = pd.cut(
            processed_data['TOTAL_AMOUNT_MPF'],
            bins=[0, value_quantiles[0.25], value_quantiles[0.5], value_quantiles[0.75], value_quantiles[0.9], float('inf')],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
            include_lowest=True
        )
        
        preprocessing_log.append(f"Created Value_Category with thresholds: {value_quantiles.to_dict()}")
    
    # 5. Create business-relevant flags
    if 'TOTAL_PRODUCT_MPF' in processed_data.columns:
        # Multi-product customer flag (important for cross-selling)
        processed_data['Multi_Product_Flag'] = processed_data['TOTAL_PRODUCT_MPF'].apply(
            lambda x: 1 if pd.to_numeric(x, errors='coerce') > 1 else 0
        )
        multi_product_pct = processed_data['Multi_Product_Flag'].mean() * 100
        preprocessing_log.append(f"Multi-product customers: {multi_product_pct:.1f}% of total")
    
    # 6. Enhance with recency information (critical for RFM analysis)
    if 'LAST_MPF_DATE' in processed_data.columns:
        reference_date = pd.Timestamp.now()
        valid_last_dates = ~processed_data['LAST_MPF_DATE'].isnull()
        
        if valid_last_dates.any():
            processed_data['Days_Since_Last_Transaction'] = np.nan
            processed_data.loc[valid_last_dates, 'Days_Since_Last_Transaction'] = (
                (reference_date - processed_data.loc[valid_last_dates, 'LAST_MPF_DATE']).dt.days
            )
            
            # Create recency categories
            processed_data['Recency_Category'] = pd.cut(
                processed_data['Days_Since_Last_Transaction'],
                bins=[0, 30, 90, 180, 365, float('inf')],
                labels=['Very Recent', 'Recent', 'Moderate', 'Lapsed', 'Inactive'],
                include_lowest=True
            )
            
            preprocessing_log.append(f"Recency statistics: Min={processed_data['Days_Since_Last_Transaction'].min():.1f}, Max={processed_data['Days_Since_Last_Transaction'].max():.1f}, Mean={processed_data['Days_Since_Last_Transaction'].mean():.1f}")
    
    # Store preprocessing log in the dataframe metadata for debugging
    processed_data._preprocessing_log = preprocessing_log
    
    # Return the enhanced data
    return processed_data

def calculate_rfm(data, recency_col, frequency_col, monetary_col):
    """
    Enhanced RFM calculation with better handling of missing values
    and additional business-relevant metrics
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Customer data
    recency_col : str
        Column name for recency (date of last transaction)
    frequency_col : str
        Column name for frequency (number of products)
    monetary_col : str
        Column name for monetary value (total amount)
    
    Returns:
    --------
    pandas.DataFrame
        RFM metrics for each customer
    """
    # Start with unique customer IDs
    rfm = data[['CUST_NO']].drop_duplicates()
    
    # Calculate Recency (days since last purchase)
    if recency_col in data.columns:
        reference_date = pd.Timestamp.now()
        
        # Group by customer and get the most recent date
        recency_data = data.groupby('CUST_NO')[recency_col].max().reset_index()
        recency_data.columns = ['CUST_NO', 'LastPurchaseDate']
        
        # Merge with RFM dataframe
        rfm = rfm.merge(recency_data, on='CUST_NO', how='left')
        
        # Calculate days since last purchase
        rfm['Recency'] = (reference_date - rfm['LastPurchaseDate']).dt.days
        
        # Identify missing values and replace with worst recency
        missing_recency = rfm['Recency'].isnull()
        if missing_recency.any():
            worst_recency = rfm['Recency'].max() * 1.5  # Worse than the worst observed
            rfm.loc[missing_recency, 'Recency'] = worst_recency
    else:
        print(f"Recency column '{recency_col}' not found in data")
        rfm['Recency'] = np.nan
    
    # Calculate Frequency (number of products)
    if frequency_col in data.columns:
        # Group by customer and get the sum of products
        freq_data = data.groupby('CUST_NO')[frequency_col].sum().reset_index()
        freq_data.columns = ['CUST_NO', 'Frequency']
        
        # Merge with RFM dataframe
        rfm = rfm.merge(freq_data, on='CUST_NO', how='left')
        
        # Handle missing values
        rfm['Frequency'] = rfm['Frequency'].fillna(1)  # Assume at least one product
        
        # Convert to numeric if needed
        rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce').fillna(1)
    else:
        print(f"Frequency column '{frequency_col}' not found in data")
        rfm['Frequency'] = 1
    
    # Calculate Monetary (total spending)
    if monetary_col in data.columns:
        # Group by customer and get the sum of spending
        monetary_data = data.groupby('CUST_NO')[monetary_col].sum().reset_index()
        monetary_data.columns = ['CUST_NO', 'Monetary']
        
        # Merge with RFM dataframe
        rfm = rfm.merge(monetary_data, on='CUST_NO', how='left')
        
        # Handle missing values
        rfm['Monetary'] = rfm['Monetary'].fillna(rfm['Monetary'].median())
        
        # Convert to numeric if needed
        rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce').fillna(rfm['Monetary'].median())
    else:
        print(f"Monetary column '{monetary_col}' not found in data")
        rfm['Monetary'] = 0
    
    # Create RFM Score for easier segmentation
    # Scale each metric to 1-5 (5 being best)
    # For Recency, lower is better so we invert the scale
    
    # Recency score (inverted - lower days = higher score)
    recency_quantiles = rfm['Recency'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    recency_quantiles = [0] + recency_quantiles + [float('inf')]
    rfm['R_Score'] = pd.cut(rfm['Recency'], bins=recency_quantiles, labels=[5, 4, 3, 2, 1], include_lowest=True, duplicates='drop').cat.codes
    
    # Frequency score
    frequency_quantiles = rfm['Frequency'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    frequency_quantiles = [0] + frequency_quantiles + [float('inf')]
    rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=frequency_quantiles, labels=[1, 2, 3, 4, 5], include_lowest=True, duplicates='drop').cat.codes
    
    # Monetary score
    monetary_quantiles = rfm['Monetary'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    monetary_quantiles = [0] + monetary_quantiles + [float('inf')]
    rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=monetary_quantiles, labels=[1, 2, 3, 4, 5], include_lowest=True, duplicates='drop').cat.codes
    
    # Combined RFM Score
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    # Customer Value Segment (based on FM score, since R is time-sensitive)
    rfm['Customer_Value'] = pd.qcut(
        rfm['F_Score'] + rfm['M_Score'], 
        q=5, 
        labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
    )
    
    # Recency Status (recent vs churned)
    rfm['Recency_Status'] = np.where(rfm['R_Score'] >= 4, 'Active', 
                             np.where(rfm['R_Score'] >= 2, 'At Risk', 'Churned'))
    
    # Loyalty Status (frequency-based)
    rfm['Loyalty_Status'] = np.where(rfm['F_Score'] >= 4, 'Loyal', 
                             np.where(rfm['F_Score'] >= 2, 'Regular', 'New/Occasional'))
    
    return rfm

def normalize_data(data, columns):
    """
    Normalize data using min-max scaling
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to normalize
    columns : list
        Columns to normalize
    
    Returns:
    --------
    pandas.DataFrame
        Normalized data
    """
    result = data.copy()
    
    for column in columns:
        if column in data.columns:
            min_val = data[column].min()
            max_val = data[column].max()
            
            # Handle the case where min and max are the same
            if min_val == max_val:
                result[column] = 0.5  # midpoint if all values are the same
            else:
                result[column] = (data[column] - min_val) / (max_val - min_val)
    
    return result

def get_cluster_info(rfm_data):
    """
    Get cluster information for business interpretation
    
    Parameters:
    -----------
    rfm_data : pandas.DataFrame
        RFM data with cluster assignments
    
    Returns:
    --------
    pandas.DataFrame
        Cluster information
    """
    # Aggregate metrics by cluster
    cluster_info = rfm_data.groupby('Cluster').agg({
        'CUST_NO': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'RFM_Score': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    cluster_info.columns = [
        'Cluster', 
        'Customer_Count', 
        'Avg_Recency_Days', 
        'Avg_Frequency', 
        'Avg_Monetary',
        'Avg_RFM_Score'
    ]
    
    # Calculate percentage of total
    total_customers = cluster_info['Customer_Count'].sum()
    cluster_info['Percentage'] = (cluster_info['Customer_Count'] / total_customers * 100).round(1)
    
    return cluster_info

def name_clusters_fixed(n_clusters=4):
    """
    Memberi nama cluster dengan nama yang tetap
    
    Parameters:
    -----------
    n_clusters : int, default=4
        Jumlah cluster
    
    Returns:
    --------
    dict
        Dictionary mapping cluster ID ke nama segmen
    """
    # Nama segmen yang ingin digunakan
    segment_names = ["Potential Loyalists", "One-Time Buyers", "Occasional Buyers", "Hibernating Customers"]
    
    # Buat dictionary untuk mapping
    cluster_names = {}
    
    # Pastikan jumlah nama sesuai dengan jumlah cluster
    if n_clusters != len(segment_names):
        # Jika tidak sama, gunakan nama yang ada dan ulangi jika perlu
        for i in range(n_clusters):
            cluster_names[i] = segment_names[i % len(segment_names)]
    else:
        # Jika sama, mapping langsung
        for i in range(n_clusters):
            cluster_names[i] = segment_names[i]
    
    return cluster_names

def generate_promo_recommendations(segmented_data, cluster_col='Cluster'):
    """
    Enhanced promo recommendation generator with business-focused strategies
    and personalized campaign recommendations
    
    Parameters:
    -----------
    segmented_data : pandas.DataFrame
        Data with customer segments/clusters
    cluster_col : str, default='Cluster'
        Column name containing cluster assignments
    
    Returns:
    --------
    pandas.DataFrame
        Promotional recommendations for each cluster
    """
    # Aggregate metrics by cluster
    cluster_metrics = segmented_data.groupby(cluster_col).agg({
        'CUST_NO': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean',
        'RFM_Score': 'mean'
    }).reset_index()
    
    # Rename for clarity
    cluster_metrics.columns = [
        'Cluster', 'Customer_Count', 'Avg_Recency_Days', 'Avg_Frequency', 
        'Avg_Monetary', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score', 
        'Avg_RFM_Score'
    ]
    
    # Get majority segments for each cluster
    segment_columns = ['Customer_Value', 'Recency_Status', 'Loyalty_Status']
    
    for segment in segment_columns:
        if segment in segmented_data.columns:
            segment_counts = segmented_data.groupby([cluster_col, segment]).size().reset_index()
            segment_counts.columns = [cluster_col, segment, 'Count']
            
            # Find the majority segment for each cluster
            majority_segments = segment_counts.loc[
                segment_counts.groupby(cluster_col)['Count'].idxmax()
            ][[cluster_col, segment]]
            
            # Merge with cluster metrics
            cluster_metrics = cluster_metrics.merge(
                majority_segments, 
                on=cluster_col,
                how='left'
            )
    
    # Add demographic insights if available
    demographic_columns = ['Usia_Kategori', 'CUST_SEX']
    for demo in demographic_columns:
        if demo in segmented_data.columns:
            demo_counts = segmented_data.groupby([cluster_col, demo]).size().reset_index()
            demo_counts.columns = [cluster_col, demo, 'Count']
            
            # Find the majority demographic for each cluster
            majority_demo = demo_counts.loc[
                demo_counts.groupby(cluster_col)['Count'].idxmax()
            ][[cluster_col, demo]]
            
            # Rename for clarity
            majority_demo.columns = [cluster_col, f'Majority_{demo}']
            
            # Merge with cluster metrics
            cluster_metrics = cluster_metrics.merge(
                majority_demo, 
                on=cluster_col,
                how='left'
            )
    
    # Define promotional strategies based on segment characteristics
    promo_strategies = []
    
    for _, row in cluster_metrics.iterrows():
        # Get cluster information
        cluster = row['Cluster']
        customer_count = row['Customer_Count']
        avg_recency = row['Avg_Recency_Days']
        avg_frequency = row['Avg_Frequency']
        avg_monetary = row['Avg_Monetary']
        
        # Get segmentation attributes if available
        customer_value = row.get('Customer_Value', 'Unknown')
        recency_status = row.get('Recency_Status', 'Unknown')
        loyalty_status = row.get('Loyalty_Status', 'Unknown')
        
        # Check if we're using fixed segment names
        if 'Segmentasi_optimal' in segmented_data.columns:
            # Get the most common segment name for this cluster
            segment_name = segmented_data[segmented_data['Cluster'] == cluster]['Segmentasi_optimal'].mode()[0]
        else:
            # Determine segment type based on cluster characteristics
            if recency_status == 'Active' and avg_frequency >= 2:
                segment_name = "Potential Loyalists"
            elif avg_frequency <= 1:
                segment_name = "One-Time Buyers"
            elif avg_recency <= 180:
                segment_name = "Occasional Buyers"
            else:
                segment_name = "Hibernating Customers"
        
        # Determine promo type based on segment
        if segment_name == "Potential Loyalists":
            promo_type = "Loyalty Rewards"
            promo_desc = "Exclusive benefits and personalized rewards to recognize loyalty and encourage advocacy"
            channel = "Personalized outreach, mobile app notifications, email"
        elif segment_name == "One-Time Buyers":
            promo_type = "Second Purchase Incentive"
            promo_desc = "Special offers to encourage a second purchase and begin building loyalty"
            channel = "Email, SMS, retargeting ads"
        elif segment_name == "Occasional Buyers":
            promo_type = "Frequency Booster"
            promo_desc = "Promotions designed to increase purchase frequency with time-limited offers"
            channel = "Email, SMS, push notifications"
        else:  # Hibernating Customers
            promo_type = "Reactivation Campaign"
            promo_desc = "Strong incentives to return with reminders of previous positive experiences"
            channel = "Email, direct mail, retargeting"
        
        # Add demographic-specific messaging if available
        demo_additions = ""
        if 'Majority_Usia_Kategori' in cluster_metrics.columns and not pd.isnull(row.get('Majority_Usia_Kategori')):
            age_category = row['Majority_Usia_Kategori']
            if '18-24' in str(age_category) or '25-34' in str(age_category):
                demo_additions += " with digital-first approach and modern messaging"
            elif '55-64' in str(age_category) or '65+' in str(age_category):
                demo_additions += " with emphasis on trust, stability and customer service"
        
        promo_desc += demo_additions
        
        promo_strategies.append({
            'Cluster': cluster,
            'Promo_Type': promo_type,
            'Promo_Description': promo_desc,
            'Channel': channel,
            'Customer_Count': customer_count,
            'Customer_Value': customer_value,
            'Recency_Status': recency_status,
            'Loyalty_Status': loyalty_status,
            'Avg_Recency_Days': avg_recency,
            'Avg_Frequency': avg_frequency,
            'Avg_Monetary': avg_monetary
        })
    
    # Convert to DataFrame
    promo_df = pd.DataFrame(promo_strategies)
    
    # Calculate budget weights based on monetary value and customer count
    promo_df['Value_Weight'] = promo_df['Avg_Monetary'] / promo_df['Avg_Monetary'].sum()
    promo_df['Count_Weight'] = promo_df['Customer_Count'] / promo_df['Customer_Count'].sum()
    promo_df['Budget_Weight'] = (promo_df['Value_Weight'] * 0.7) + (promo_df['Count_Weight'] * 0.3)
    
    return promo_df

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
        'LAST_MPF_DATE': ['2024-12-01', '2024-10-20', '2025-02-15', '2023-08-10'],
        'MAX_MPF_AMOUNT': [3000000, 2000000, 4000000, 1500000],
        'MIN_MPF_AMOUNT': [1000000, 1000000, 1500000, 500000],
        'LAST_MPF_AMOUNT': [2000000, 1000000, 3000000, 1000000],
        'LAST_MPF_INST': [12, 6, 18, 9],
        'LAST_MPF_TOP': [24, 12, 36, 18],
        'AVG_MPF_INST': [10, 5, 15, 8],
        'PRINCIPAL': [3000000, 2000000, 5000000, 1000000],
        'GRS_DP': [500000, 400000, 600000, 200000],
        'JMH_CON_SBLM_MPF': [1, 0, 2, 0],
        'JMH_PPC': [3, 1, 5, 0]
    })

    return data
