import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from utils.data_utils import generate_promo_recommendations

def show_promo_mapping_page():
    """
    Fungsi untuk menampilkan halaman pemetaan promo
    """
    st.markdown('<p class="section-title">Promo Mapping & Campaign Planning</p>', unsafe_allow_html=True)
    
    if not st.session_state.segmentation_completed:
        st.warning("Please complete the segmentation analysis first.")
        return
    
    # Load data hasil segmentasi
    segmented_data = st.session_state.segmented_data
    
    if segmented_data is None:
        st.error("Segmentation data not found. Please redo the segmentation analysis.")
        return
    
    st.markdown("### Customize Promo Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        campaign_name = st.text_input("Campaign Name", "SPEKTRA Mid-Year Promotion")
        start_date = st.date_input("Campaign Start Date", datetime.now())
    
    with col2:
        budget = st.number_input("Total Campaign Budget (Rp)", min_value=10000000, value=100000000, step=10000000)
        end_date = st.date_input("Campaign End Date", datetime.now().replace(month=datetime.now().month+1))
    
    st.markdown("### Promo Strategy")
    
    # Options for strategies
    strategy_options = {
        "value": "Value-based (High to Low Value)",
        "recency": "Recency-based (Most Recent First)",
        "mixed": "Mixed Approach (Balanced)"
    }
    
    strategy = st.radio("Select Promo Allocation Strategy", options=list(strategy_options.keys()), 
                       format_func=lambda x: strategy_options[x])
    
    # Button untuk membuat rekomendasi
    if st.button("Generate Promo Recommendations"):
        with st.spinner("Generating promo recommendations..."):
            try:
                # Generate promo recommendations
                promo_df = generate_promo_recommendations(segmented_data, cluster_col='Cluster')
                
                # Add campaign details
                promo_df['Campaign_Name'] = campaign_name
                promo_df['Start_Date'] = start_date
                promo_df['End_Date'] = end_date
                
                # Calculate budget allocation based on strategy
                if strategy == "value":
                    # Allocate based on monetary value
                    promo_df['Budget_Percentage'] = promo_df['Avg_Monetary'] / promo_df['Avg_Monetary'].sum()
                elif strategy == "recency":
                    # Allocate based on recency (lower is better)
                    promo_df['Recency_Score'] = 1 / (promo_df['Avg_Recency_Days'] + 1)
                    promo_df['Budget_Percentage'] = promo_df['Recency_Score'] / promo_df['Recency_Score'].sum()
                else:  # mixed
                    # Use a balanced approach
                    promo_df['Mixed_Score'] = (
                        (1 / (promo_df['Avg_Recency_Days'] + 1)) * 0.4 + 
                        (promo_df['Avg_Monetary'] / promo_df['Avg_Monetary'].max()) * 0.4 +
                        (promo_df['Avg_Frequency'] / promo_df['Avg_Frequency'].max()) * 0.2
                    )
                    promo_df['Budget_Percentage'] = promo_df['Mixed_Score'] / promo_df['Mixed_Score'].sum()
                
                # Calculate actual budget allocation
                promo_df['Allocated_Budget'] = promo_df['Budget_Percentage'] * budget
                
                # Store in session state
                st.session_state.promo_mapped_data = promo_df
                st.session_state.promo_mapping_completed = True
                
                # Display results
                display_promo_results(promo_df, budget, start_date, end_date)
                
            except Exception as e:
                st.error(f"Error generating promo recommendations: {e}")
                st.exception(e)  # This will display the full traceback
                st.warning("Please check your segmentation data and try again.")

def display_promo_results(promo_df, budget, start_date, end_date):
    """
    Menampilkan hasil pemetaan promo
    
    Parameters:
    -----------
    promo_df : pandas.DataFrame
        Data hasil pemetaan promo
    budget : float
        Total budget kampanye
    start_date : datetime
        Tanggal mulai kampanye
    end_date : datetime
        Tanggal berakhir kampanye
    """
    st.markdown("### Promo Mapping Results")
    
    # Overview card
    st.markdown("""
    <div class="box-container">
        <h4 class="card-title">Campaign Overview</h4>
        <p><strong>Duration:</strong> {start_date} to {end_date} ({days} days)</p>
        <p><strong>Total Budget:</strong> Rp {budget:,.0f}</p>
        <p><strong>Target Clusters:</strong> {n_clusters}</p>
        <p><strong>Total Customers:</strong> {n_customers:,}</p>
    </div>
    """.format(
        start_date=start_date.strftime("%d %b %Y"),
        end_date=end_date.strftime("%d %b %Y"),
        days=(end_date - start_date).days,
        budget=budget,
        n_clusters=promo_df.shape[0],
        n_customers=promo_df['Customer_Count'].sum()
    ), unsafe_allow_html=True)
    
    # Tampilkan tabel pemetaan promo
    st.markdown("#### Cluster Promo Mapping")
    
    # Format currency dan percentage
    promo_df['Allocated_Budget'] = promo_df['Allocated_Budget'].apply(lambda x: f"Rp {x:,.0f}")
    promo_df['Budget_Percentage'] = promo_df['Budget_Percentage'].apply(lambda x: f"{x*100:.1f}%")
    
    # Tampilkan tabel
    st.dataframe(promo_df[[
        'Cluster', 'Customer_Value', 'Recency_Status', 'Loyalty_Status',
        'Customer_Count', 'Promo_Type', 'Channel', 'Budget_Percentage', 'Allocated_Budget'
    ]])
    
    # Budget allocation visualization
    st.markdown("#### Budget Allocation by Cluster")
    
    # Convert back to numeric for visualization
    promo_df['Allocated_Budget_Num'] = promo_df['Allocated_Budget'].apply(
        lambda x: float(x.replace("Rp ", "").replace(",", ""))
    )
    
    # Create pie chart
    fig = px.pie(
        promo_df, 
        values='Allocated_Budget_Num', 
        names=[f"Cluster {c}" for c in promo_df['Cluster']],
        title="Budget Allocation by Cluster",
        color_discrete_sequence=px.colors.sequential.Blues,
        hover_data=['Customer_Count', 'Promo_Type']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Distribution by Value Segment
    st.markdown("#### Customer Distribution by Segment")
    
    fig = px.bar(
        promo_df, 
        x='Customer_Value',
        y='Customer_Count',
        color='Recency_Status',
        title="Customer Distribution by Value and Recency",
        text='Customer_Count',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Promo details for each cluster
    st.markdown("#### Detailed Promo Plans by Cluster")
    
    for i, row in promo_df.iterrows():
        with st.expander(f"Cluster {row['Cluster']} - {row['Customer_Value']} / {row['Recency_Status']} Customers"):
            st.markdown(f"""
            <div class="box-container">
                <h4 class="card-title">{row['Promo_Type']}</h4>
                <p><strong>Description:</strong> {row['Promo_Description']}</p>
                <p><strong>Target Audience:</strong> {row['Customer_Count']:,} customers ({row['Customer_Value']}, {row['Recency_Status']}, {row['Loyalty_Status']})</p>
                <p><strong>Marketing Channels:</strong> {row['Channel']}</p>
                <p><strong>Budget Allocation:</strong> {row['Allocated_Budget']} ({row['Budget_Percentage']})</p>
                <p><strong>Average Spending:</strong> Rp {row['Avg_Monetary']:,.0f} per customer</p>
                <p><strong>Avg Products Owned:</strong> {row['Avg_Frequency']:.1f} products</p>
                <p><strong>Days Since Last Transaction:</strong> {row['Avg_Recency_Days']:.0f} days</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Download link
    csv = promo_df.to_csv(index=False)
    st.download_button(
        label="Download Promo Mapping",
        data=csv,
        file_name="promo_mapping.csv",
        mime="text/csv"
    )
    
    # Next steps
    st.info("You can now proceed to the Dashboard to get a comprehensive view of your customer segments and promo strategies.")