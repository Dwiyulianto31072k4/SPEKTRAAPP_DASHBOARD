import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

def show_dashboard_page():
    """
    Fungsi untuk menampilkan halaman dashboard
    """
    st.markdown('<p class="section-title">SPEKTRA Customer Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Check if segmentation and promo mapping are completed
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
    else:
        st.info("Segmentation analysis has not been completed yet. Some dashboard components are not available.")
    
    # Check if promo mapping is completed
    if st.session_state.promo_mapping_completed and st.session_state.promo_mapped_data is not None:
        promo_data = st.session_state.promo_mapped_data
        campaign_overview(promo_data)
    else:
        st.info("Promo mapping has not been completed yet. Campaign metrics are not available.")

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

def campaign_overview(promo_data):
    """
    Menampilkan overview kampanye
    
    Parameters:
    -----------
    promo_data : pandas.DataFrame
        Data hasil pemetaan promo
    """
    st.markdown("### Campaign & Promo Overview")
    
    # Convert budget to numeric for visualization
    if 'Allocated_Budget' in promo_data.columns and isinstance(promo_data['Allocated_Budget'].iloc[0], str):
        promo_data['Allocated_Budget_Num'] = promo_data['Allocated_Budget'].apply(
            lambda x: float(x.replace("Rp ", "").replace(",", ""))
        )
    elif 'Allocated_Budget' in promo_data.columns:
        promo_data['Allocated_Budget_Num'] = promo_data['Allocated_Budget']
    
    # Key metrics
    total_customers = promo_data['Customer_Count'].sum() if 'Customer_Count' in promo_data.columns else 0
    total_budget = promo_data['Allocated_Budget_Num'].sum() if 'Allocated_Budget_Num' in promo_data.columns else 0
    avg_budget_per_customer = total_budget / total_customers if total_customers > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <p style="font-size: 14px; margin-bottom: 0;">Target Customers</p>
            <h2 style="margin-top: 0;">{:,}</h2>
        </div>
        """.format(total_customers), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <p style="font-size: 14px; margin-bottom: 0;">Total Campaign Budget</p>
            <h2 style="margin-top: 0;">Rp {:,.0f}</h2>
        </div>
        """.format(total_budget), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <p style="font-size: 14px; margin-bottom: 0;">Budget per Customer</p>
            <h2 style="margin-top: 0;">Rp {:,.0f}</h2>
        </div>
        """.format(avg_budget_per_customer), unsafe_allow_html=True)
    
    # Promotion Types Overview
    st.markdown("#### Promotion Types Overview")
    
    if 'Promo_Type' in promo_data.columns:
        # Budget allocation by promo type
        promo_budget = promo_data.groupby('Promo_Type').agg({
            'Allocated_Budget_Num': 'sum',
            'Customer_Count': 'sum'
        }).reset_index()
        
        # Calculate percentage
        promo_budget['Budget_Percentage'] = promo_budget['Allocated_Budget_Num'] / promo_budget['Allocated_Budget_Num'].sum() * 100
        
        # Create combined visualization
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=("Budget Allocation by Promo Type", "Customers by Promo Type")
        )
        
        # Budget allocation pie chart
        fig.add_trace(
            go.Pie(
                labels=promo_budget['Promo_Type'],
                values=promo_budget['Allocated_Budget_Num'],
                textinfo='percent',
                hole=0.4,
                marker_colors=px.colors.qualitative.Pastel,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Customer distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=promo_budget['Promo_Type'],
                values=promo_budget['Customer_Count'],
                textinfo='percent',
                hole=0.4,
                marker_colors=px.colors.qualitative.Pastel,
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Promotion type data not available.")
    
    # Customer Value and Recency Status
    st.markdown("#### Customer Segments in Campaign")
    
    if 'Customer_Value' in promo_data.columns and 'Recency_Status' in promo_data.columns:
        # Create combined data
        value_recency = promo_data.groupby(['Customer_Value', 'Recency_Status']).agg({
            'Customer_Count': 'sum',
            'Allocated_Budget_Num': 'sum'
        }).reset_index()
        
        # Stacked bar chart
        fig = px.bar(
            value_recency,
            x='Customer_Value',
            y='Customer_Count',
            color='Recency_Status',
            title="Customer Distribution by Value and Recency Status",
            text_auto=True,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            xaxis_title="Customer Value Segment",
            yaxis_title="Number of Customers",
            legend_title="Recency Status",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Budget allocation heatmap
        value_recency_pivot = value_recency.pivot_table(
            index='Customer_Value', 
            columns='Recency_Status', 
            values='Allocated_Budget_Num',
            aggfunc='sum'
        ).fillna(0)
        
        # Format for display (convert to millions)
        value_recency_pivot_display = value_recency_pivot / 1000000
        
        fig = px.imshow(
            value_recency_pivot_display,
            text_auto='.2f',
            title="Budget Allocation (in millions) by Customer Segment",
            color_continuous_scale='Blues',
            labels=dict(x="Recency Status", y="Customer Value", color="Budget (Millions)")
        )
        
        fig.update_layout(height=350)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Customer segment data not available.")

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
