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
            
        # Add recency segments for business use
        rfm['Recency_Segment'] = pd.cut(
            rfm['Recency'],
            bins=[0, 30, 90, 180, 365, float('inf')],
            labels=['Very Recent', 'Recent', 'Moderate', 'Lapsed', 'Inactive'],
            include_lowest=True
        )
    else:
        print(f"Recency column '{recency_col}' not found in data")
        rfm['Recency'] = np.nan
        rfm['Recency_Segment'] = 'Unknown'
    
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
        
        # Add frequency segments for business use
        rfm['Frequency_Segment'] = pd.cut(
            rfm['Frequency'],
            bins=[0, 1, 2, 3, 5, float('inf')],
            labels=['Single', 'Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
    else:
        print(f"Frequency column '{frequency_col}' not found in data")
        rfm['Frequency'] = 1
        rfm['Frequency_Segment'] = 'Unknown'
    
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
        
        # Add monetary segments for business use
        monetary_quantiles = rfm['Monetary'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        monetary_quantiles = [0] + monetary_quantiles + [float('inf')]
        
        rfm['Monetary_Segment'] = pd.cut(
            rfm['Monetary'],
            bins=monetary_quantiles,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
    else:
        print(f"Monetary column '{monetary_col}' not found in data")
        rfm['Monetary'] = 0
        rfm['Monetary_Segment'] = 'Unknown'
    
    # Create RFM Score for easier segmentation
    # Scale each metric to 1-5 (5 being best)
    # For Recency, lower is better so we invert the scale
    
    # Recency score (inverted - lower days = higher score)
    recency_quantiles = rfm['Recency'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    recency_quantiles = [0] + recency_quantiles + [float('inf')]
    # recency_quantiles.reverse()  # Removed to fix pd.cut monotonicity error  # Reverse for inverted scoring
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
    
    # Calculate CLV (Customer Lifetime Value) proxy
    # Using recency as a retention probability factor
    # Retention probability decreases as recency increases
    retention_factor = 1 / (1 + rfm['Recency']/365)
    avg_annual_value = rfm['Monetary'] * (rfm['Frequency'] / 12)  # Approximating annual value
    predicted_years = 3  # Prediction window
    discount_rate = 0.1  # Annual discount rate
    
    # Simple CLV calculation
    rfm['CLV'] = avg_annual_value * sum([(retention_factor / (1 + discount_rate))**i for i in range(1, predicted_years+1)])
    
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
    
    # Determine cluster characteristics
    cluster_info['Recency_Level'] = cluster_info['Avg_Recency_Days'].apply(
        lambda x: 'High' if x <= 30 else 'Medium' if x <= 90 else 'Low'
    )
    
    cluster_info['Frequency_Level'] = cluster_info['Avg_Frequency'].apply(
        lambda x: 'High' if x >= 3 else 'Medium' if x >= 2 else 'Low'
    )
    
    cluster_info['Monetary_Level'] = cluster_info['Avg_Monetary'].apply(
        lambda x: 'High' if x >= cluster_info['Avg_Monetary'].quantile(0.75) 
                 else 'Medium' if x >= cluster_info['Avg_Monetary'].quantile(0.25) 
                 else 'Low'
    )
    
    # Create descriptive cluster names based on RFM characteristics
    def create_cluster_name(row):
        r_level = row['Recency_Level']
        f_level = row['Frequency_Level']
        m_level = row['Monetary_Level']
        
        if r_level == 'High' and f_level == 'High' and m_level == 'High':
            return 'Champions'
        elif r_level == 'High' and m_level == 'High':
            return 'High-Value Recent Customers'
        elif f_level == 'High' and m_level == 'High':
            return 'Loyal High-Spenders'
        elif r_level == 'High' and f_level == 'High':
            return 'Loyal Recent Customers'
        elif m_level == 'High':
            return 'Big Spenders'
        elif r_level == 'High':
            return 'Recently Active'
        elif f_level == 'High':
            return 'Frequent Buyers'
        elif r_level == 'Low' and f_level == 'Low' and m_level == 'Low':
            return 'At Risk'
        else:
            return 'Regular Customers'
    
    cluster_info['Cluster_Name'] = cluster_info.apply(create_cluster_name, axis=1)
    
    # Sort by RFM Score (highest first)
    cluster_info = cluster_info.sort_values('Avg_RFM_Score', ascending=False)
    
    return cluster_info

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
        'RFM_Score': 'mean',
        'CLV': 'mean'
    }).reset_index()
    
    # Rename for clarity
    cluster_metrics.columns = [
        'Cluster', 'Customer_Count', 'Avg_Recency_Days', 'Avg_Frequency', 
        'Avg_Monetary', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score', 
        'Avg_RFM_Score', 'Avg_CLV'
    ]
    
    # Get majority segments for each cluster
    segment_columns = ['Customer_Value', 'Recency_Status', 'Loyalty_Status']
    
    for segment in segment_columns:
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
        # Determine appropriate promo type based on customer characteristics
        customer_value = row['Customer_Value']
        recency_status = row['Recency_Status']
        loyalty_status = row['Loyalty_Status']
        
        # Champions (high value + active + loyal)
        if customer_value in ['Diamond', 'Platinum'] and recency_status == 'Active' and loyalty_status == 'Loyal':
            promo_type = "VIP Loyalty Rewards"
            promo_desc = "Exclusive benefits and personalized rewards to recognize loyalty and encourage advocacy"
            channel = "Personalized outreach, VIP events, dedicated account manager"
            
        # High Value but At Risk (high value + not active)
        elif customer_value in ['Diamond', 'Platinum'] and recency_status != 'Active':
            promo_type = "Win-back Premium Offer"
            promo_desc = "High-value reactivation incentives with personalized outreach to recover valuable customers"
            channel = "Direct contact, personalized email/SMS, special offers"
            
        # Active but Low Value (active + low value)
        elif recency_status == 'Active' and customer_value in ['Bronze', 'Silver']:
            promo_type = "Upsell Campaign"
            promo_desc = "Targeted product recommendations to increase spend with special upgrade offers"
            channel = "App notifications, email, targeted ads"
            
        # Loyal but Decreasing Value (loyal + at risk)
        elif loyalty_status == 'Loyal' and recency_status == 'At Risk':
            promo_type = "Loyalty Reinforcement"
            promo_desc = "Reminders of loyalty benefits and special incentives to re-engage"
            channel = "Email, SMS, loyalty program communications"
            
        # New Customer Nurturing (new/occasional + active)
        elif loyalty_status == 'New/Occasional' and recency_status == 'Active':
            promo_type = "New Customer Development"
            promo_desc = "Onboarding sequence and introductory offers to encourage repeat purchase"
            channel = "Email journey, app notifications, educational content"
            
        # Dormant Customer Reactivation (churned + was valuable)
        elif recency_status == 'Churned' and customer_value in ['Gold', 'Platinum', 'Diamond']:
            promo_type = "Reactivation Campaign"
            promo_desc = "Strong incentives to return with reminders of previous positive experiences"
            channel = "Direct mail, email, retargeting ads"
            
        # Low Value Churned (churned + low value)
        elif recency_status == 'Churned' and customer_value in ['Bronze', 'Silver']:
            promo_type = "Final Chance Offer"
            promo_desc = "Last attempt to re-engage with special offer before reducing marketing investment"
            channel = "Email, SMS"
            
        # Regular Customer Maintenance (regular + any recency)
        else:
            promo_type = "Engagement Program"
            promo_desc = "Regular touchpoints and offers to maintain relationship and encourage increased engagement"
            channel = "Email, SMS, app notifications"
        
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
            'Cluster': row['Cluster'],
            'Promo_Type': promo_type,
            'Promo_Description': promo_desc,
            'Channel': channel,
            'Customer_Count': row['Customer_Count'],
            'Customer_Value': row['Customer_Value'],
            'Recency_Status': row['Recency_Status'],
            'Loyalty_Status': row['Loyalty_Status'],
            'Avg_Recency_Days': row['Avg_Recency_Days'],
            'Avg_Frequency': row['Avg_Frequency'],
            'Avg_Monetary': row['Avg_Monetary'],
            'Avg_CLV': row.get('Avg_CLV', 0)
        })
    
    # Convert to DataFrame
    promo_df = pd.DataFrame(promo_strategies)
    
# Calculate optimal budget allocation based on CLV or customer count
    if 'Avg_CLV' in promo_df.columns and not promo_df['Avg_CLV'].isnull().all():
        # Calculate total CLV per cluster (Avg CLV × Customer Count)
        promo_df['Total_CLV'] = promo_df['Avg_CLV'] * promo_df['Customer_Count']
        
        # Allocate budget proportionally to total CLV
        promo_df['Budget_Weight'] = promo_df['Total_CLV'] / promo_df['Total_CLV'].sum()
    else:
        # Fallback to allocation based on customer count and monetary value
        promo_df['Value_Weight'] = promo_df['Avg_Monetary'] / promo_df['Avg_Monetary'].sum()
        promo_df['Count_Weight'] = promo_df['Customer_Count'] / promo_df['Customer_Count'].sum()
        promo_df['Budget_Weight'] = (promo_df['Value_Weight'] * 0.7) + (promo_df['Count_Weight'] * 0.3)
    
    return promo_df

def predict_campaign_roi(promo_df, budget):
    """
    Predicts potential return on investment for promotional campaigns
    
    Parameters:
    -----------
    promo_df : pandas.DataFrame
        DataFrame with promotional recommendations by segment
    budget : float
        Total campaign budget
    
    Returns:
    --------
    pandas.DataFrame
        Enhanced DataFrame with ROI predictions
    """
    result_df = promo_df.copy()
    
    # Allocate budget based on weights
    result_df['Allocated_Budget'] = result_df['Budget_Weight'] * budget
    result_df['Budget_Per_Customer'] = result_df['Allocated_Budget'] / result_df['Customer_Count']
    
    # Estimate response rates based on customer characteristics
    # These values should be calibrated based on historical campaign performance
    
    # Base response rate
    result_df['Base_Response_Rate'] = 0.05  # 5% baseline
    
    # Adjust for recency
    recency_factors = {
        'Active': 1.5,    # 50% higher than baseline
        'At Risk': 0.8,   # 20% lower than baseline
        'Churned': 0.3    # 70% lower than baseline
    }
    
    result_df['Recency_Factor'] = result_df['Recency_Status'].map(recency_factors).fillna(1.0)
    
    # Adjust for customer value
    value_factors = {
        'Diamond': 1.3,   # 30% higher than baseline
        'Platinum': 1.2,  # 20% higher than baseline
        'Gold': 1.1,      # 10% higher than baseline
        'Silver': 0.9,    # 10% lower than baseline
        'Bronze': 0.8     # 20% lower than baseline
    }
    
    result_df['Value_Factor'] = result_df['Customer_Value'].map(value_factors).fillna(1.0)
    
    # Adjust for loyalty
    loyalty_factors = {
        'Loyal': 1.3,          # 30% higher than baseline
        'Regular': 1.0,         # Baseline
        'New/Occasional': 0.7   # 30% lower than baseline
    }
    
    result_df['Loyalty_Factor'] = result_df['Loyalty_Status'].map(loyalty_factors).fillna(1.0)
    
    # Calculate adjusted response rate
    result_df['Predicted_Response_Rate'] = (
        result_df['Base_Response_Rate'] * 
        result_df['Recency_Factor'] * 
        result_df['Value_Factor'] * 
        result_df['Loyalty_Factor']
    ).clip(0.001, 0.5)  # Cap between 0.1% and 50%
    
    # Predict number of responders
    result_df['Predicted_Responders'] = (
        result_df['Customer_Count'] * result_df['Predicted_Response_Rate']
    ).round(0).astype(int)
    
    # Predict average purchase value based on historical average
    result_df['Predicted_Purchase_Value'] = result_df['Avg_Monetary'] * 0.8  # Assuming 80% of historical avg
    
    # Calculate predicted revenue
    result_df['Predicted_Revenue'] = result_df['Predicted_Responders'] * result_df['Predicted_Purchase_Value']
    
    # Calculate ROI
    result_df['Predicted_ROI'] = (result_df['Predicted_Revenue'] - result_df['Allocated_Budget']) / result_df['Allocated_Budget']
    result_df['ROI_Percent'] = result_df['Predicted_ROI'] * 100
    
    # Calculate overall campaign metrics
    total_budget = result_df['Allocated_Budget'].sum()
    total_revenue = result_df['Predicted_Revenue'].sum()
    overall_roi = (total_revenue - total_budget) / total_budget
    
    # Add overall metrics to each row for reference
    result_df['Campaign_Budget'] = total_budget
    result_df['Campaign_Revenue'] = total_revenue
    result_df['Campaign_ROI'] = overall_roi
    
    return result_df

def create_rfm_3d_visualization(segmented_data, cluster_col='Cluster', title="Customer Segmentation 3D Visualization"):
    """
    Creates an interactive 3D visualization of RFM clusters
    
    Parameters:
    -----------
    segmented_data : pandas.DataFrame
        Data with RFM metrics and cluster assignments
    cluster_col : str, default='Cluster'
        Column name containing cluster assignments
    title : str
        Chart title
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot
    """
    import plotly.graph_objects as go
    
    # Create a sample of data if very large (for performance)
    if len(segmented_data) > 5000:
        # Stratified sampling to preserve cluster proportions
        sample_data = segmented_data.groupby(cluster_col).apply(
            lambda x: x.sample(min(1000, len(x)), random_state=42)
        ).reset_index(drop=True)
    else:
        sample_data = segmented_data.copy()
    
    # Create hover text with customer details
    sample_data['hover_text'] = (
        "Customer: " + sample_data['CUST_NO'].astype(str) + "<br>" +
        "Recency: " + sample_data['Recency'].round(0).astype(int).astype(str) + " days<br>" +
        "Frequency: " + sample_data['Frequency'].round(1).astype(str) + "<br>" +
        "Monetary: Rp " + sample_data['Monetary'].apply(lambda x: f"{x:,.0f}")
    )
    
    # Add customer value and status if available
    if 'Customer_Value' in sample_data.columns:
        sample_data['hover_text'] += "<br>Value: " + sample_data['Customer_Value'].astype(str)
    
    if 'Recency_Status' in sample_data.columns:
        sample_data['hover_text'] += "<br>Status: " + sample_data['Recency_Status'].astype(str)
    
    # Create the 3D scatter plot
    fig = go.Figure()
    
    # Add a trace for each cluster
    for cluster in sorted(sample_data[cluster_col].unique()):
        cluster_data = sample_data[sample_data[cluster_col] == cluster]
        
        # Determine a good name for the cluster
        if 'Customer_Value' in cluster_data.columns and 'Recency_Status' in cluster_data.columns:
            value = cluster_data['Customer_Value'].mode()[0]
            status = cluster_data['Recency_Status'].mode()[0]
            cluster_name = f"Cluster {cluster}: {value}/{status}"
        else:
            cluster_name = f"Cluster {cluster}"
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['Recency'],
            y=cluster_data['Frequency'],
            z=cluster_data['Monetary'],
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.7
            ),
            name=cluster_name,
            hovertext=cluster_data['hover_text'],
            hoverinfo='text'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Recency (days)',
            yaxis_title='Frequency (products)',
            zaxis_title='Monetary (Rp)'
        ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.7)'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )
    
    return fig


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

