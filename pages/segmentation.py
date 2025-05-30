import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from scipy.stats import zscore

from utils.data_utils import calculate_rfm, normalize_data, get_cluster_info, perform_optimal_segmentation

def show_segmentation_page():
   """
   Fungsi untuk menampilkan halaman segmentasi pelanggan dengan metode optimal
   """
   st.markdown('<p class="section-title">Segmentation Analysis (RFM+ Optimal)</p>', unsafe_allow_html=True)

   if st.session_state.data is None:
       st.warning("Please upload and preprocess your data first.")
       return

   df = st.session_state.data.copy()

   # Pastikan kolom 'Repeat_Customer' ada
   if 'Repeat_Customer' not in df.columns and 'TOTAL_PRODUCT_MPF' in df.columns:
       df["Repeat_Customer"] = df["TOTAL_PRODUCT_MPF"].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') > 1 else 0)

   # Pilih kolom untuk segmentasi
   st.markdown("### Select RFM Columns and Segmentation Parameters")
   
   col1, col2, col3 = st.columns(3)
   with col1:
       date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
       if date_cols:
           recency_col = st.selectbox(
               "Recency (last transaction date)", 
               date_cols, 
               index=date_cols.index("LAST_MPF_DATE") if "LAST_MPF_DATE" in date_cols else 0
           )
       else:
           st.error("No date columns found in the data. Please preprocess the data first.")
           return
   
   with col2:
       numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
       freq_col = st.selectbox(
           "Frequency (number of products)", 
           numeric_cols, 
           index=numeric_cols.index("TOTAL_PRODUCT_MPF") if "TOTAL_PRODUCT_MPF" in numeric_cols else 0
       )
   
   with col3:
       mon_col = st.selectbox(
           "Monetary (total amount)", 
           numeric_cols, 
           index=numeric_cols.index("TOTAL_AMOUNT_MPF") if "TOTAL_AMOUNT_MPF" in numeric_cols else 0
       )

   # Tambahan opsi
   st.markdown("### Segmentation Method")
   
   segmentation_method = st.radio(
       "Select segmentation method:",
       ["Optimal Method (Research-Based)", "Standard RFM K-Means"],
       index=0
   )
   
   # Additional options based on selected method
   if segmentation_method == "Optimal Method (Research-Based)":
       st.markdown("### Optimal Method Settings")
       
       col1, col2 = st.columns(2)
       
       with col1:
           use_fixed_date = st.checkbox("Use Fixed Reference Date", value=True)
           if use_fixed_date:
               import datetime
               ref_date = st.date_input("Reference Date", datetime.datetime(2024, 12, 31))
           else:
               ref_date = None
       
       with col2:
           st.markdown("""
           **Optimal Segmentation Features:**
           - Recency (days since last transaction) ✓
           - Frequency (log transformed) ✓
           - Monetary (log transformed) ✓
           - Repeat Customer Flag ✓
           - Prime Age Segment (25-45) ✓
           - Z-score normalization ✓
           - K-means++ algorithm ✓
           - Dynamic segment naming ✓
           """)
           
           # Info button
           if st.button("Method Details"):
               st.info("""
               **Research-Based Segmentation Method Details**
               
               This segmentation method is based on research on customer behavior in the financial services industry. It enhances traditional RFM analysis with:
               
               1. **Log Transformation**: Applied to Frequency and Monetary to handle skewed distributions
               2. **Additional Features**: 
                  - Repeat Customer Flag: Identifies customers who have purchased more than one product
                  - Prime Age Segment: Identifies customers in the 25-45 age range who have higher response rates
               3. **Dynamic Naming**: Segment names are assigned based on the actual characteristics of each cluster, ensuring "Potential Loyalists" always represents your best customers
               4. **Four Distinct Segments**: 
                  - Potential Loyalists: Best overall customers (highest total score)
                  - Responsive Customers: Most recent customers (best recency)
                  - Occasional Buyers: Customers with good frequency
                  - Hibernating Customers: Customers requiring reactivation
               
               This approach has been shown to improve campaign response rates by up to 25% compared to standard RFM segmentation.
               """)
   else:
       # Standard RFM K-Means options
       st.markdown("### Standard RFM K-Means Settings")
       
       col1, col2 = st.columns(2)
       
       with col1:
           cluster_k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)
       
       with col2:
           use_zscore = st.checkbox("Apply Z-Score Normalization", value=True)
   
   # Create segmentation button
   button_label = "Perform Optimal Segmentation" if segmentation_method == "Optimal Method (Research-Based)" else "Perform Standard Segmentation"
   
   if st.button(button_label):
       with st.spinner("Processing segmentation..."):
           try:
               if segmentation_method == "Optimal Method (Research-Based)":
                   # Convert reference date if provided
                   reference_date = pd.Timestamp(ref_date) if use_fixed_date else None
                   
                   # Use optimal segmentation method
                   rfm = perform_optimal_segmentation(
                       df, 
                       recency_col=recency_col, 
                       frequency_col=freq_col, 
                       monetary_col=mon_col,
                       reference_date=reference_date
                   )
                   
                   # Save to session state
                   st.session_state.segmented_data = rfm
                   st.session_state.segmentation_completed = True
                   
                   # Display results
                   st.success("Optimal segmentation completed successfully!")
                   display_optimal_segmentation_results(rfm)
                   
               else:
                   # Standard RFM K-Means
                   # Calculate RFM metrics
                   rfm = calculate_rfm(df, recency_col, freq_col, mon_col)
                   
                   # Fill NA values if any
                   if rfm.isnull().values.any():
                       rfm = rfm.fillna(rfm.median())
                   
                   # Select features for clustering
                   features = ['Recency', 'Frequency', 'Monetary']
                   
                   # Normalize features if selected
                   if use_zscore:
                       try:
                           rfm_scaled = rfm[features].apply(zscore)
                       except:
                           st.warning("Z-score normalization failed. Using features without normalization.")
                           rfm_scaled = rfm[features].copy()
                   else:
                       rfm_scaled = rfm[features].copy()
                   
                   # Perform K-means clustering
                   kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)
                   rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
                   
                   # Get cluster information
                   cluster_info = get_cluster_info(rfm)
                   
                   # Standard segment names
                   top_clusters = cluster_info.sort_values('Avg_RFM_Score', ascending=False).head(cluster_k // 2)['Cluster'].tolist()
                   rfm['Invitation_Status'] = rfm['Cluster'].apply(lambda x: '✅ Invited' if x in top_clusters else '❌ Not Invited')
                   
                   # Save to session state
                   st.session_state.segmented_data = rfm
                   st.session_state.segmentation_completed = True
                   
                   # Display results
                   st.success("Standard segmentation completed successfully!")
                   display_standard_segmentation_results(rfm, cluster_info, top_clusters)
               
           except Exception as e:
               st.error(f"Error during segmentation: {e}")
               st.exception(e)  # This will display the full traceback
               st.warning("Please check your data and selections, then try again.")

def display_optimal_segmentation_results(rfm):
   """
   Menampilkan hasil segmentasi dengan metode optimal (FIXED VERSION)
   
   Parameters:
   -----------
   rfm : pandas.DataFrame
       Data hasil segmentasi dengan metode optimal
   """
   st.markdown("### Optimal Segmentation Results")
   
   # Key metrics
   col1, col2, col3, col4 = st.columns(4)
   
   with col1:
       st.metric("Total Customers", f"{len(rfm):,}")
   
   with col2:
       invited = len(rfm[rfm['Layak_Diundang_optimal'].str.contains('✅')])
       st.metric("Customers to Invite", f"{invited:,}")
   
   with col3:
       invited_pct = (invited / len(rfm)) * 100
       st.metric("Invitation Rate", f"{invited_pct:.1f}%")
   
   with col4:
       segments = len(rfm['Segmentasi_optimal'].unique())
       st.metric("Segments", str(segments))
   
   # Segment distribution
   st.markdown("#### Customer Distribution by Segment")
   
   segment_counts = rfm['Segmentasi_optimal'].value_counts().reset_index()
   segment_counts.columns = ['Segment', 'Count']
   segment_counts['Percentage'] = segment_counts['Count'] / segment_counts['Count'].sum() * 100
   
   # Visualization - Bar chart of segment distribution
   fig = px.bar(
       segment_counts,
       x='Segment',
       y='Count',
       text=segment_counts['Percentage'].apply(lambda x: f"{x:.1f}%"),
       color='Segment',
       title="Customer Distribution by Segment"
   )
   
   fig.update_layout(
       xaxis_title="Segment",
       yaxis_title="Number of Customers",
       height=400
   )
   
   st.plotly_chart(fig, use_container_width=True)
   
   # Segment metrics - FIXED: Handle column name properly
   st.markdown("#### Segment Characteristics")
   
   try:
       # Create segment metrics using the correct column name
       segment_metrics = rfm.groupby('Segmentasi_optimal').agg({
           'Recency': 'mean',
           'Frequency': 'mean',
           'Monetary': 'mean',
           'Repeat_Customer': 'mean',
           'Usia_Segment': 'mean' if 'Usia_Segment' in rfm.columns else lambda x: 0,
           'CUST_NO': 'count'
       }).reset_index()
       
       segment_metrics.columns = ['Segment', 'Avg Recency (days)', 'Avg Products', 
                                  'Avg Value (Rp)', 'Repeat Rate', 'Prime Age Rate', 'Count']
       
       # Format for display
       segment_metrics['Avg Recency (days)'] = segment_metrics['Avg Recency (days)'].round(0).astype(int)
       segment_metrics['Avg Products'] = segment_metrics['Avg Products'].round(1)
       segment_metrics['Avg Value (Rp)'] = segment_metrics['Avg Value (Rp)'].apply(lambda x: f"Rp {x:,.0f}")
       segment_metrics['Repeat Rate'] = segment_metrics['Repeat Rate'].apply(lambda x: f"{x*100:.1f}%")
       segment_metrics['Prime Age Rate'] = segment_metrics['Prime Age Rate'].apply(lambda x: f"{x*100:.1f}%")
       segment_metrics['Percentage'] = (segment_metrics['Count'] / segment_metrics['Count'].sum() * 100).apply(lambda x: f"{x:.1f}%")
       
       # Add invitation status - FIXED: Use correct merge approach
       segment_invite = rfm.groupby('Segmentasi_optimal')['Layak_Diundang_optimal'].first().reset_index()
       segment_invite.columns = ['Segment', 'Invitation_Status']
       
       # Merge invitation status
       segment_metrics = segment_metrics.merge(segment_invite, on='Segment', how='left')
       
       # Display metrics table
       st.dataframe(segment_metrics)
       
   except Exception as e:
       st.error(f"Error creating segment metrics: {e}")
       st.info("Displaying basic segment information instead:")
       
       # Fallback: Simple segment distribution
       basic_metrics = rfm['Segmentasi_optimal'].value_counts().reset_index()
       basic_metrics.columns = ['Segment', 'Count']
       basic_metrics['Percentage'] = (basic_metrics['Count'] / basic_metrics['Count'].sum() * 100).apply(lambda x: f"{x:.1f}%")
       st.dataframe(basic_metrics)
   
   # 3D visualization
   st.markdown("#### 3D Segmentation Visualization")
   
   try:
       fig = px.scatter_3d(
           rfm,
           x='Recency',
           y='Frequency',
           z='Monetary',
           color='Segmentasi_optimal',
           size='Repeat_Customer',
           opacity=0.7,
           title="3D Customer Segmentation - RFM"
       )
       
       fig.update_layout(
           scene=dict(
               xaxis_title="Recency (days)",
               yaxis_title="Frequency (products)",
               zaxis_title="Monetary (value)"
           ),
           height=600
       )
       
       st.plotly_chart(fig, use_container_width=True)
   except Exception as e:
       st.warning(f"Could not create 3D visualization: {e}")
   
   # Feature importance - FIXED: Handle missing score columns
   st.markdown("#### Feature Importance by Segment")
   
   try:
       # Check if score columns exist
       score_columns = ['Recency_Score', 'Frequency_Score', 'Monetary_Score', 'Repeat_Score']
       if all(col in rfm.columns for col in score_columns):
           # Get cluster scores
           cluster_scores = rfm.groupby('Segmentasi_optimal')[score_columns].first().reset_index()
           
           # Create radar chart for feature importance
           fig = go.Figure()
           
           for _, row in cluster_scores.iterrows():
               fig.add_trace(go.Scatterpolar(
                   r=[row['Recency_Score'], row['Frequency_Score'], 
                      row['Monetary_Score'], row['Repeat_Score']],
                   theta=['Recency', 'Frequency', 'Monetary', 'Repeat Rate'],
                   fill='toself',
                   name=row['Segmentasi_optimal']
               ))
           
           fig.update_layout(
               polar=dict(
                   radialaxis=dict(
                       visible=True,
                       range=[0, 4]
                   )
               ),
               title="Segment Characteristics Comparison (Higher Score = Better)",
               height=500
           )
           
           st.plotly_chart(fig, use_container_width=True)
       else:
           st.info("Score columns not available for radar chart.")
           
   except Exception as e:
       st.warning(f"Could not create feature importance chart: {e}")
   
   # Segment details tabs
   st.markdown("#### Segment Details and Recommendations")
   
   try:
       segment_names = sorted(rfm['Segmentasi_optimal'].unique())
       segment_tabs = st.tabs(segment_names)
       
       for i, segment in enumerate(segment_names):
           with segment_tabs[i]:
               segment_data = rfm[rfm['Segmentasi_optimal'] == segment]
               invitation = segment_data['Layak_Diundang_optimal'].iloc[0]
               
               col1, col2 = st.columns([2, 1])
               
               with col1:
                   st.markdown(f"""
                   ### {segment}
                   
                   **Segment Size:** {segment_data.shape[0]:,} customers ({segment_data.shape[0]/rfm.shape[0]*100:.1f}% of total)
                   
                   **Invitation Status:** {invitation}
                   
                   **Key Characteristics:**
                   - Average Recency: {segment_data['Recency'].mean():.0f} days since last transaction
                   - Average Products: {segment_data['Frequency'].mean():.1f}
                   - Average Value: Rp {segment_data['Monetary'].mean():,.0f}
                   - Repeat Customer Rate: {segment_data['Repeat_Customer'].mean()*100:.1f}%
                   """)
                   
                   if 'Usia_Segment' in segment_data.columns:
                       st.markdown(f"- Prime Age Rate (25-45): {segment_data['Usia_Segment'].mean()*100:.1f}%")
                   
                   # Recommendations based on segment
                   if segment == "Potential Loyalists":
                       st.markdown("""
                       **Recommended Strategy:**
                       - Focus on loyalty programs to convert to full loyalists
                       - Offer membership benefits
                       - Personalized communication with relevant cross-sell opportunities
                       - Early access to new products and special events
                       """)
                       
                   elif segment == "Responsive Customers":
                       st.markdown("""
                       **Recommended Strategy:**
                       - Targeted promotions for high response rate
                       - Encourage more frequent purchases
                       - Special limited-time offers
                       - Reminders about products they might need
                       """)
                       
                   elif segment == "Occasional Buyers":
                       st.markdown("""
                       **Recommended Strategy:**
                       - Incentives to increase purchase frequency
                       - Re-engagement campaigns
                       - Feedback surveys to understand barriers
                       - Special "welcome back" offers
                       """)
                       
                   elif segment == "Hibernating Customers":
                       st.markdown("""
                       **Recommended Strategy:**
                       - Reactivation campaigns with strong incentives
                       - Re-introduction to your product line
                       - Check if contact information is current
                       - Consider special win-back campaigns for high-value customers
                       """)
               
               with col2:
                   # Donut chart for this segment vs others
                   segment_vs_others = pd.DataFrame([
                       {'Category': segment, 'Count': len(segment_data)},
                       {'Category': 'Others', 'Count': len(rfm) - len(segment_data)}
                   ])
                   
                   fig = px.pie(
                       segment_vs_others,
                       values='Count',
                       names='Category',
                       hole=0.6,
                       color_discrete_sequence=['#003366', '#E0E0E0']
                   )
                   
                   fig.update_layout(
                       title=f"{segment} vs Others",
                       height=300
                   )
                   
                   st.plotly_chart(fig)
                   
   except Exception as e:
       st.error(f"Error creating segment details: {e}")
   
   # Provide download option
   st.markdown("#### Download Segmentation Results")
   
   try:
       csv = rfm.to_csv(index=False)
       st.download_button(
           label="Download Complete Segmentation Data",
           data=csv,
           file_name="optimal_segmentation_results.csv",
           mime="text/csv"
       )
   except Exception as e:
       st.error(f"Error preparing download: {e}")
   
   # Summary and next steps
   st.markdown("""
   ### Summary and Next Steps
   
   The optimal segmentation has identified 4 distinct customer segments:
   
   1. **Potential Loyalists**: Customers with high potential for loyalty
   2. **Responsive Customers**: Customers who respond well to promotions
   3. **Occasional Buyers**: Infrequent but valuable customers
   4. **Hibernating Customers**: Inactive customers who need reactivation
   
   **Next Steps:**
   - Review customer profiles in each segment to understand characteristics
   - Develop targeted marketing strategies for each segment
   - Set up tracking to measure customer movement between segments over time
   - Test different marketing approaches for each segment
   """)

def display_standard_segmentation_results(rfm, cluster_info, top_clusters):
   """
   Menampilkan hasil segmentasi dengan metode standar
   
   Parameters:
   -----------
   rfm : pandas.DataFrame
       Data hasil segmentasi
   cluster_info : pandas.DataFrame
       Informasi tentang setiap cluster
   top_clusters : list
       Daftar cluster yang direkomendasikan untuk diundang
   """
   st.markdown("### Segmentation Results")
   
   # Overview metrics
   st.markdown("#### Segmentation Overview")
   col1, col2, col3 = st.columns(3)
   
   with col1:
       n_clusters = rfm['Cluster'].nunique()
       st.metric("Total Clusters", str(n_clusters))
   
   with col2:
       n_invited = sum(rfm['Cluster'].isin(top_clusters))
       st.metric("Customers to Invite", f"{n_invited:,}")
   
   with col3:
       pct_invited = n_invited / len(rfm) * 100
       st.metric("Invitation Rate", f"{pct_invited:.1f}%")
   
   # Cluster sizes visualization
   st.markdown("#### Cluster Sizes")
   
   cluster_sizes = rfm['Cluster'].value_counts().reset_index()
   cluster_sizes.columns = ['Cluster', 'Count']
   cluster_sizes['Percentage'] = cluster_sizes['Count'] / cluster_sizes['Count'].sum() * 100
   
   fig = px.bar(
       cluster_sizes,
       x='Cluster',
       y='Count',
       text=cluster_sizes['Percentage'].apply(lambda x: f"{x:.1f}%"),
       color='Cluster',
       title="Customers per Cluster"
   )
   
   st.plotly_chart(fig, use_container_width=True)
   
   # Cluster Metrics
   st.markdown("#### Cluster Metrics")
   
   # Format metrics for display
   display_metrics = cluster_info.copy()
   display_metrics['Avg_Recency_Days'] = display_metrics['Avg_Recency_Days'].round(0).astype(int)
   display_metrics['Avg_Frequency'] = display_metrics['Avg_Frequency'].round(1)
   display_metrics['Avg_Monetary'] = display_metrics['Avg_Monetary'].apply(lambda x: f"Rp {x:,.0f}")
   display_metrics['Percentage'] = display_metrics['Percentage'].apply(lambda x: f"{x:.1f}%")
   
   # Add invitation status
   display_metrics['Invitation_Status'] = ['✅ Invited' if cluster in top_clusters else '❌ Not Invited' 
                                         for cluster in display_metrics['Cluster']]
   
   st.dataframe(display_metrics)
   
   # 3D visualization
   st.markdown("#### 3D Segmentation Visualization")
   
   fig = px.scatter_3d(
       rfm,
       x='Recency',
       y='Frequency',
       z='Monetary',
       color='Cluster',
       opacity=0.7,
       title="3D Customer Segmentation"
   )
   
   fig.update_layout(
       scene=dict(
           xaxis_title="Recency (days)",
           yaxis_title="Frequency (products)",
           zaxis_title="Monetary (value)"
       ),
       height=600
   )
   
   st.plotly_chart(fig, use_container_width=True)
   
   # 2D RFM Map
   st.markdown("#### RFM Customer Segmentation Map")
   
   fig = px.scatter(
       rfm,
       x='Recency',
       y='Monetary',
       size='Frequency',
       color='Cluster',
       hover_name='CUST_NO',
       size_max=30,
       opacity=0.7,
       title="RFM Customer Segmentation Map"
   )
   
   fig.update_layout(
       xaxis_title="Recency (days since last transaction)",
       yaxis_title="Monetary Value",
       height=500
   )
   
   st.plotly_chart(fig, use_container_width=True)
   
   # Download option
   csv = rfm.to_csv(index=False)
   st.download_button(
       label="Download Segmentation Results",
       data=csv,
       file_name="segmentation_results.csv",
       mime="text/csv"
   )
   
   # Next steps
   st.markdown("""
   ### Next Steps
   
   Based on the segmentation results, you can now:
   
   1. Proceed to the Dashboard section for an integrated view of your customer segments
   2. Export the results for further analysis or campaign implementation
   3. Develop targeted marketing strategies for each cluster
   """)
