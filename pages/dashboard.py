import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

def show_dashboard_page():
    """
    Fungsi untuk menampilkan halaman dashboard (tanpa promo mapping)
    """
    st.markdown('<p class="section-title">SPEKTRA Customer Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Check if EDA is completed
    if not st.session_state.eda_completed:
        st.warning("Please complete data preprocessing and EDA first.")
        return
    
    data = st.session_state.data
    
    if data is None:
        st.error("Data not found. Please upload and preprocess data first.")
        return
    
    # Customer Overview
    customer_overview(data)
    
    # Check if segmentation is completed
    if st.session_state.segmentation_completed and st.session_state.segmented_data is not None:
        segmented_data = st.session_state.segmented_data
        segmentation_overview(segmented_data)
        invitation_summary(segmented_data)
    else:
        st.info("Segmentation analysis has not been completed yet. Complete the segmentation to see detailed customer segments and invitation recommendations.")

def customer_overview(data):
    """
    Menampilkan overview pelanggan
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    """
    st.markdown("### Customer Overview")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <p style="font-size: 14px; margin-bottom: 0;">Total Customers</p>
            <h2 style="margin-top: 0;">{:,}</h2>
        </div>
        """.format(data['CUST_NO'].nunique()), unsafe_allow_html=True)
    
    with col2:
        # Calculate average transaction value
        if 'TOTAL_AMOUNT_MPF' in data.columns:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(data['TOTAL_AMOUNT_MPF']):
                data['TOTAL_AMOUNT_MPF'] = pd.to_numeric(data['TOTAL_AMOUNT_MPF'], errors='coerce')
            
            avg_transaction = data['TOTAL_AMOUNT_MPF'].mean()
            st.markdown("""
            <div class="metric-box">
                <p style="font-size: 14px; margin-bottom: 0;">Avg Transaction Value</p>
                <h2 style="margin-top: 0;">Rp {:,.0f}</h2>
            </div>
            """.format(avg_transaction), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-box">
                <p style="font-size: 14px; margin-bottom: 0;">Avg Transaction Value</p>
                <h2 style="margin-top: 0;">N/A</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Calculate percentage of multi-product customers
        if 'TOTAL_PRODUCT_MPF' in data.columns:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(data['TOTAL_PRODUCT_MPF']):
                data['TOTAL_PRODUCT_MPF'] = pd.to_numeric(data['TOTAL_PRODUCT_MPF'], errors='coerce')
            
            multi_product_pct = (data['TOTAL_PRODUCT_MPF'] > 1).mean() * 100
            st.markdown("""
            <div class="metric-box">
                <p style="font-size: 14px; margin-bottom: 0;">Multi-Product Customers</p>
                <h2 style="margin-top: 0;">{:.1f}%</h2>
            </div>
            """.format(multi_product_pct), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-box">
                <p style="font-size: 14px; margin-bottom: 0;">Multi-Product Customers</p>
                <h2 style="margin-top: 0;">N/A</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Calculate average customer age
        if 'Usia' in data.columns:
            avg_age = data['Usia'].mean()
            st.markdown("""
            <div class="metric-box">
                <p style="font-size: 14px; margin-bottom: 0;">Average Customer Age</p>
                <h2 style="margin-top: 0;">{:.1f} years</h2>
            </div>
            """.format(avg_age), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-box">
                <p style="font-size: 14px; margin-bottom: 0;">Average Customer Age</p>
                <h2 style="margin-top: 0;">N/A</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Product and demographic distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Product Category Distribution")
        
        if 'MPF_CATEGORIES_TAKEN' in data.columns:
            category_counts = data['MPF_CATEGORIES_TAKEN'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            fig = px.pie(
                category_counts, 
                values='Count', 
                names='Category',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product category data not available.")
    
    with col2:
        st.markdown("#### Customer Demographics")
        
        if 'Usia_Kategori' in data.columns:
            age_counts = data['Usia_Kategori'].value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
            
            # Gender distribution if available
            if 'CUST_SEX' in data.columns:
                # Create a combined chart
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "pie"}, {"type": "pie"}]],
                    subplot_titles=("Age Distribution", "Gender Distribution")
                )
                
                # Add age distribution
                fig.add_trace(
                    go.Pie(
                        labels=age_counts['Age Group'],
                        values=age_counts['Count'],
                        hole=0.4,
                        marker_colors=px.colors.sequential.Blues,
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add gender distribution
                gender_counts = data['CUST_SEX'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']
                gender_labels = gender_counts['Gender'].map({'M': 'Male', 'F': 'Female'})
                
                fig.add_trace(
                    go.Pie(
                        labels=gender_labels,
                        values=gender_counts['Count'],
                        hole=0.4,
                        marker_colors=['#003366', '#66b3ff'],
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Just show age distribution
                fig = px.pie(
                    age_counts, 
                    values='Count', 
                    names='Age Group',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Demographic data not available.")

def segmentation_overview(segmented_data):
    """
    Menampilkan overview segmentasi
    
    Parameters:
    -----------
    segmented_data : pandas.DataFrame
        Data hasil segmentasi
    """
    st.markdown("### Segmentation Overview")
    
    # Number of clusters
    n_clusters = segmented_data['Cluster'].nunique()
    
    # Create cluster metrics
    cluster_metrics = segmented_data.groupby('Cluster').agg({
        'CUST_NO': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    cluster_metrics.columns = ['Cluster', 'Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
    
    # Create cluster names based on characteristics
    cluster_metrics['Cluster_Name'] = cluster_metrics.apply(
        lambda row: create_cluster_name(row['Avg_Recency'], row['Avg_Frequency'], row['Avg_Monetary']),
        axis=1
    )
    
    # Calculate percentages
    cluster_metrics['Percentage'] = cluster_metrics['Count'] / cluster_metrics['Count'].sum() * 100
    
    # Cluster distribution visualization
    st.markdown("#### Customer Distribution Across Segments")
    
    fig = px.bar(
        cluster_metrics,
        x='Cluster',
        y='Count',
        color='Cluster_Name',
        text=cluster_metrics['Percentage'].apply(lambda x: f"{x:.1f}%"),
        title="Customer Count by Cluster",
        hover_data=['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary'],
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Customers",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics visualization
    st.markdown("#### Cluster Characteristics")
    
    # Normalize data for radar chart
    radar_data = cluster_metrics.copy()
    
    # Reverse Recency since lower is better
    max_recency = radar_data['Avg_Recency'].max()
    radar_data['Recency_Normalized'] = 1 - (radar_data['Avg_Recency'] / max_recency)
    
    # Normalize Frequency and Monetary
    radar_data['Frequency_Normalized'] = radar_data['Avg_Frequency'] / radar_data['Avg_Frequency'].max()
    radar_data['Monetary_Normalized'] = radar_data['Avg_Monetary'] / radar_data['Avg_Monetary'].max()
    
    # Create radar chart
    fig = go.Figure()
    
    for i, row in radar_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Recency_Normalized'], row['Frequency_Normalized'], row['Monetary_Normalized']],
            theta=['Recency', 'Frequency', 'Monetary'],
            fill='toself',
            name=f"Cluster {row['Cluster']}: {row['Cluster_Name']}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bubble chart untuk visualisasi RFM
    st.markdown("#### RFM Segmentation Map")
    
    fig = px.scatter(
        segmented_data,
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
    
    # Table with cluster information
    st.markdown("#### Cluster Details")
    
    # Format the metrics
    display_metrics = cluster_metrics.copy()
    display_metrics['Avg_Recency'] = display_metrics['Avg_Recency'].round(0).astype(int)
    display_metrics['Avg_Frequency'] = display_metrics['Avg_Frequency'].round(1)
    display_metrics['Avg_Monetary'] = display_metrics['Avg_Monetary'].apply(lambda x: f"Rp {x:,.0f}")
    display_metrics['Percentage'] = display_metrics['Percentage'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_metrics[['Cluster', 'Cluster_Name', 'Count', 'Percentage', 
                                 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']])

def invitation_summary(segmented_data):
    """
    Menampilkan ringkasan undangan berdasarkan segmentasi
    
    Parameters:
    -----------
    segmented_data : pandas.DataFrame
        Data hasil segmentasi
    """
    st.markdown("### Invitation Summary")
    
    # Check which invitation column exists
    invitation_col = None
    if 'Layak_Diundang_optimal' in segmented_data.columns:
        invitation_col = 'Layak_Diundang_optimal'
    elif 'Invitation_Status' in segmented_data.columns:
        invitation_col = 'Invitation_Status'
    
    if invitation_col is None:
        st.warning("No invitation status found in segmentation data.")
        return
    
    # Calculate invitation metrics
    total_customers = len(segmented_data)
    invited_customers = len(segmented_data[segmented_data[invitation_col].str.contains('✅', na=False)])
    not_invited_customers = total_customers - invited_customers
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <p style="font-size: 14px; margin-bottom: 0;">Total Customers</p>
            <h2 style="margin-top: 0;">{:,}</h2>
        </div>
        """.format(total_customers), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #28a745; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <p style="font-size: 14px; margin-bottom: 0;">✅ Invited</p>
            <h2 style="margin-top: 0;">{:,} ({:.1f}%)</h2>
        </div>
        """.format(invited_customers, invited_customers/total_customers*100), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #dc3545; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <p style="font-size: 14px; margin-bottom: 0;">❌ Not Invited</p>
            <h2 style="margin-top: 0;">{:,} ({:.1f}%)</h2>
        </div>
        """.format(not_invited_customers, not_invited_customers/total_customers*100), unsafe_allow_html=True)
    
    # Invitation status by segment
    st.markdown("#### Invitation Status by Segment")
    
    # Check if optimal segmentation was used
    if 'Segmentasi_optimal' in segmented_data.columns:
        segment_col = 'Segmentasi_optimal'
    else:
        segment_col = 'Cluster'
    
    # Create invitation summary by segment
    invitation_summary = segmented_data.groupby([segment_col, invitation_col]).size().unstack(fill_value=0)
    invitation_summary['Total'] = invitation_summary.sum(axis=1)
    
    # Calculate percentages
    for col in invitation_summary.columns[:-1]:  # Exclude 'Total' column
        invitation_summary[f'{col}_pct'] = invitation_summary[col] / invitation_summary['Total'] * 100
    
    # Display as a chart
    invited_counts = []
    not_invited_counts = []
    segment_names = []
    
    for segment in invitation_summary.index:
        segment_names.append(str(segment))
        
        invited = 0
        not_invited = 0
        
        for col in invitation_summary.columns:
            if '✅' in str(col):
                invited = invitation_summary.loc[segment, col]
            elif '❌' in str(col):
                not_invited = invitation_summary.loc[segment, col]
        
        invited_counts.append(invited)
        not_invited_counts.append(not_invited)
    
    # Create stacked bar chart
    fig = go.Figure(data=[
        go.Bar(name='✅ Invited', x=segment_names, y=invited_counts, marker_color='#28a745'),
        go.Bar(name='❌ Not Invited', x=segment_names, y=not_invited_counts, marker_color='#dc3545')
    ])
    
    fig.update_layout(
        barmode='stack',
        title='Invitation Status by Customer Segment',
        xaxis_title='Customer Segment',
        yaxis_title='Number of Customers',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed table
    st.markdown("#### Detailed Invitation Breakdown")
    st.dataframe(invitation_summary)
    
    # Recommendations
    st.markdown("#### Recommendations")
    
    if invited_customers > 0:
        st.success(f"""
        **Action Items:**
        - Focus marketing efforts on {invited_customers:,} invited customers ({invited_customers/total_customers*100:.1f}% of total)
        - Prioritize segments with high invitation rates for campaign execution
        - Consider re-engagement strategies for non-invited customers to improve their segment scores
        """)
    else:
        st.warning("No customers are currently recommended for invitation. Consider adjusting segmentation criteria.")

def create_cluster_name(recency, frequency, monetary):
    """
    Create descriptive name for a cluster based on RFM values
    
    Parameters:
    -----------
    recency : float
        Average recency value
    frequency : float
        Average frequency value
    monetary : float
        Average monetary value
    
    Returns:
    --------
    str
        Descriptive name for the cluster
    """
    # Define thresholds (these can be adjusted based on data)
    recency_threshold = 90  # days
    frequency_threshold = 1.5  # transactions
    monetary_threshold = 5000000  # Rp
    
    # Determine characteristics
    is_recent = recency <= recency_threshold
    is_frequent = frequency >= frequency_threshold
    is_high_value = monetary >= monetary_threshold
    
    # Create name based on characteristics
    if is_recent and is_frequent and is_high_value:
        return "Champions"
    elif is_recent and is_high_value:
        return "High-Value Recent"
    elif is_frequent and is_high_value:
        return "Loyal High-Spenders"
    elif is_recent and is_frequent:
        return "Loyal Recent"
    elif is_high_value:
        return "Big Spenders"
    elif is_recent:
        return "Recently Active"
    elif is_frequent:
        return "Frequent Buyers"
    else:
        return "At Risk"
