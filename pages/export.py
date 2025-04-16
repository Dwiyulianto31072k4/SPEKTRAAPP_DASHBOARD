import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import json
import os
import datetime

def show_export_page():
    """
    Fungsi untuk menampilkan halaman ekspor dan dokumentasi
    """
    st.markdown('<p class="section-title">Export & Documentation</p>', unsafe_allow_html=True)
    
    # Check for data
    if st.session_state.data is None:
        st.warning("Please upload and preprocess data first.")
        return
    
    # Get data from session state
    data = st.session_state.data
    segmented_data = st.session_state.segmented_data if st.session_state.segmentation_completed else None
    promo_data = st.session_state.promo_mapped_data if st.session_state.promo_mapping_completed else None
    
    # Show export options
    st.markdown("### Export Options")
    
    tab1, tab2, tab3 = st.tabs(["Export Data", "Export Report", "Export Charts"])
    
    with tab1:
        show_data_export(data, segmented_data, promo_data)
    
    with tab2:
        show_report_export(data, segmented_data, promo_data)
    
    with tab3:
        show_chart_export(data, segmented_data, promo_data)
    
    # Show documentation
    st.markdown("### Documentation")
    show_documentation()

def show_data_export(data, segmented_data, promo_data):
    """
    Tampilkan opsi ekspor data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
    promo_data : pandas.DataFrame or None
        Data hasil pemetaan promo
    """
    st.markdown("#### Export Data to CSV/Excel")
    
    # Data selection
    export_option = st.radio("Select data to export:", 
                            ["Processed Customer Data", "Segmentation Results", "Promo Mapping Results", "All Data"])
    
    # Export format
    export_format = st.radio("Select export format:", ["CSV", "Excel"])
    
    # Show preview based on selection
    if export_option == "Processed Customer Data":
        st.dataframe(data.head())
        export_df = data
    elif export_option == "Segmentation Results":
        if segmented_data is None:
            st.warning("Segmentation data not available. Please complete the segmentation first.")
            return
        st.dataframe(segmented_data.head())
        export_df = segmented_data
    elif export_option == "Promo Mapping Results":
        if promo_data is None:
            st.warning("Promo mapping data not available. Please complete the promo mapping first.")
            return
        st.dataframe(promo_data.head())
        export_df = promo_data
    else:  # All Data
        # Create a list of DataFrames to export
        export_dfs = {"processed_data": data}
        
        if segmented_data is not None:
            export_dfs["segmented_data"] = segmented_data
        
        if promo_data is not None:
            export_dfs["promo_data"] = promo_data
    
    # Export button
    if export_option != "All Data":
        if export_format == "CSV":
            csv = export_df.to_csv(index=False)
            filename = f"{export_option.lower().replace(' ', '_')}.csv"
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
            )
        else:  # Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, sheet_name='Data', index=False)
            
            filename = f"{export_option.lower().replace(' ', '_')}.xlsx"
            
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        if export_format == "CSV":
            st.warning("For exporting all data, only Excel format is supported with multiple sheets.")
        else:  # Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                data.to_excel(writer, sheet_name='Processed_Data', index=False)
                
                if segmented_data is not None:
                    segmented_data.to_excel(writer, sheet_name='Segmentation', index=False)
                
                if promo_data is not None:
                    promo_data.to_excel(writer, sheet_name='Promo_Mapping', index=False)
            
            st.download_button(
                label="Download All Data (Excel)",
                data=output.getvalue(),
                file_name="spektra_all_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

def show_report_export(data, segmented_data, promo_data):
    """
    Tampilkan opsi ekspor laporan
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
    promo_data : pandas.DataFrame or None
        Data hasil pemetaan promo
    """
    st.markdown("#### Generate Analytical Report")
    
    # Report options
    report_type = st.selectbox("Select report type:", 
                              ["Customer Segmentation Report", "Campaign Strategy Report", "Complete Analysis Report"])
    
    # Report details
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name", "FIFGROUP")
        report_title = st.text_input("Report Title", f"SPEKTRA {report_type}")
    
    with col2:
        author = st.text_input("Author/Team", "Data Science Team")
        date = st.date_input("Report Date", datetime.datetime.now())
    
    # Check prerequisites for report generation
    if report_type in ["Customer Segmentation Report", "Complete Analysis Report"] and segmented_data is None:
        st.warning("Segmentation data required for this report. Please complete the segmentation first.")
        can_generate = False
    elif report_type in ["Campaign Strategy Report", "Complete Analysis Report"] and promo_data is None:
        st.warning("Promo mapping data required for this report. Please complete the promo mapping first.")
        can_generate = False
    else:
        can_generate = True
    
    if can_generate and st.button("Generate Report"):
        with st.spinner("Generating report..."):
            # Build report content
            report_content = generate_report_content(
                report_type, data, segmented_data, promo_data,
                company_name, report_title, author, date
            )
            
            # Export as HTML
            html_report = report_content
            
            # Provide download link
            filename = f"{report_type.lower().replace(' ', '_')}_{date.strftime('%Y%m%d')}.html"
            
            # Encode report content as base64
            b64 = base64.b64encode(html_report.encode()).decode()
            
            # Create download link
            href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Preview
            with st.expander("Preview Report"):
                st.components.v1.html(html_report, height=600)

def show_chart_export(data, segmented_data, promo_data):
    """
    Tampilkan opsi ekspor chart
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
    promo_data : pandas.DataFrame or None
        Data hasil pemetaan promo
    """
    st.markdown("#### Export Visualizations")
    
    # Available charts
    chart_options = []
    
    # Basic charts (always available)
    if 'MPF_CATEGORIES_TAKEN' in data.columns:
        chart_options.append("Product Category Distribution")
    
    if 'Usia_Kategori' in data.columns:
        chart_options.append("Age Distribution")
    
    if 'CUST_SEX' in data.columns:
        chart_options.append("Gender Distribution")
    
    # Segmentation charts
    if segmented_data is not None:
        chart_options.append("RFM Customer Segmentation")
        chart_options.append("Cluster Distribution")
    
    # Promo charts
    if promo_data is not None:
        chart_options.append("Budget Allocation by Cluster")
        chart_options.append("Customer Value Segments")
    
    if not chart_options:
        st.warning("No charts available for export. Please complete at least the EDA section.")
        return
    
    # Let user select charts
    selected_charts = st.multiselect("Select charts to export:", chart_options)
    
    if not selected_charts:
        st.info("Please select at least one chart to export.")
        return
    
    # Chart format
    chart_format = st.radio("Export format:", ["PNG", "SVG", "HTML"])
    
    if st.button("Export Charts"):
        with st.spinner("Preparing charts for export..."):
            # Placeholder for chart export function
            for chart in selected_charts:
                # Here we would generate the charts based on selection
                st.info(f"Chart export for '{chart}' is not implemented in this demo.")
            
            st.success("All charts have been prepared for export.")

def show_documentation():
    """
    Tampilkan dokumentasi
    """
    st.markdown("#### SPEKTRA Customer Segmentation & Promo App Documentation")
    
    with st.expander("About This Application"):
        st.markdown("""
        The SPEKTRA Customer Segmentation & Promo App is a comprehensive analytics tool designed to help
        marketing teams segment customers and create targeted promotional campaigns.
        
        Key features include:
        - Data preprocessing and cleaning
        - Exploratory data analysis
        - RFM (Recency, Frequency, Monetary) analysis
        - K-means clustering for customer segmentation
        - Promotional campaign mapping
        - Interactive dashboards
        - Report and data export
        
        This application was developed by the Data Science Team at FIFGROUP.
        """)
    
    with st.expander("How to Use This App"):
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **Upload & Preprocessing**
           - Upload your customer data Excel file
           - or use the example data for demonstration
           - The app will automatically detect date columns and preprocess the data
        
        2. **Exploratory Data Analysis (EDA)**
           - Explore distributions of key variables
           - Analyze customer demographics
           - Study transaction patterns
        
        3. **Segmentation Analysis**
           - Select RFM columns and clustering parameters
           - Perform K-means clustering for customer segmentation
           - Analyze the resulting clusters
        
        4. **Promo Mapping**
           - Define campaign parameters
           - Select a campaign strategy
           - Generate promotional recommendations for each cluster
        
        5. **Dashboard**
           - View an integrated dashboard of all analysis
           - Get key insights about your customer segments
        
        6. **Export & Documentation**
           - Export data as CSV or Excel
           - Generate analytical reports
           - Export visualizations
        """)
    
    with st.expander("Data Requirements"):
        st.markdown("""
        ### Required Data Columns
        
        For optimal functionality, your customer data should include:
        
        **Customer Identifiers:**
        - `CUST_NO`: Unique customer ID
        
        **Transaction Information:**
        - `LAST_MPF_DATE`: Date of last transaction
        - `TOTAL_PRODUCT_MPF`: Total number of products purchased
        - `TOTAL_AMOUNT_MPF`: Total monetary value of transactions
        
        **Additional Useful Fields:**
        - `MPF_CATEGORIES_TAKEN`: Product categories
        - `BIRTH_DATE`: Customer birth date
        - `CUST_SEX`: Customer gender
        - `EDU_TYPE`: Education level
        - `MARITAL_STAT`: Marital status
        
        Note: The app can still function with minimal data, but will provide more insights with comprehensive data.
        """)

def generate_report_content(report_type, data, segmented_data, promo_data, 
                           company_name, report_title, author, date):
    """
    Generate report content based on type
    
    Parameters:
    -----------
    report_type : str
        Type of report to generate
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
    promo_data : pandas.DataFrame or None
        Data hasil pemetaan promo
    company_name : str
        Nama perusahaan
    report_title : str
        Judul laporan
    author : str
        Penulis/tim
    date : datetime.date
        Tanggal laporan
    
    Returns:
    --------
    str
        Report content as HTML
    """
    # Format date
    date_str = date.strftime("%d %B %Y")
    
    # Begin HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 1px solid #003366;
                padding-bottom: 20px;
            }}
            h1 {{
                color: #003366;
                margin-bottom: 10px;
            }}
            h2 {{
                color: #003366;
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }}
            h3 {{
                color: #003366;
            }}
            .metadata {{
                color: #666;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #003366;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .chart-placeholder {{
                background-color: #f9f9f9;
                padding: 20px;
                text-align: center;
                border: 1px solid #ddd;
                margin-bottom: 20px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{report_title}</h1>
            <div class="metadata">
                <p>Prepared for: {company_name}</p>
                <p>Prepared by: {author}</p>
                <p>Date: {date_str}</p>
            </div>
        </div>
    """
    
    # Add specific content based on report type
    if report_type == "Customer Segmentation Report":
        html_content += generate_segmentation_report_content(data, segmented_data)
    elif report_type == "Campaign Strategy Report":
        html_content += generate_campaign_report_content(data, segmented_data, promo_data)
    else:  # Complete Analysis Report
        html_content += generate_complete_report_content(data, segmented_data, promo_data)
    
    # Add footer and close HTML
    html_content += """
        <div class="footer">
            <p>Generated by SPEKTRA Customer Segmentation & Promo App</p>
            <p>© FIFGROUP Data Science Team</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_segmentation_report_content(data, segmented_data):
    """
    Generate content for segmentation report
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame
        Data hasil segmentasi
    
    Returns:
    --------
    str
        Report content as HTML
    """
    # Basic customer statistics
    total_customers = data['CUST_NO'].nunique()
    avg_age = data['Usia'].mean() if 'Usia' in data.columns else "N/A"
    
    # Cluster statistics
    n_clusters = segmented_data['Cluster'].nunique()
    
    # HTML content
    html_content = f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report presents the results of customer segmentation analysis based on RFM methodology
            (Recency, Frequency, Monetary). {total_customers:,} unique customers were analyzed and grouped
            into {n_clusters} distinct segments.</p>
            
            <p>The segmentation provides strategic insights into customer behavior patterns and allows
            for targeted marketing approaches tailored to each segment's characteristics.</p>
        </div>
        
        <div class="section">
            <h2>Customer Overview</h2>
            <p>Total Customers: {total_customers:,}</p>
            <p>Average Customer Age: {avg_age:.1f} years</p>
            
            <h3>Key Customer Demographics</h3>
            <div class="chart-placeholder">
                [Customer Demographics Visualization]
            </div>
        </div>
        
        <div class="section">
            <h2>Segmentation Methodology</h2>
            <p>The segmentation was performed using the RFM (Recency, Frequency, Monetary) framework combined with K-means clustering:</p>
            <ul>
                <li><strong>Recency:</strong> How recently a customer made a transaction</li>
                <li><strong>Frequency:</strong> How often a customer makes transactions</li>
                <li><strong>Monetary:</strong> How much money a customer spends</li>
            </ul>
            <p>K-means clustering with {n_clusters} clusters was used to group similar customers based on their normalized RFM values.</p>
        </div>
        
        <div class="section">
            <h2>Segment Profiles</h2>
            <div class="chart-placeholder">
                [Cluster Visualization]
            </div>
    """
    
    # Add cluster information
    cluster_stats = segmented_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CUST_NO': 'count'
    }).reset_index()
    
    # Add segment descriptions
    for _, row in cluster_stats.iterrows():
        cluster = row['Cluster']
        count = row['CUST_NO']
        percentage = count / total_customers * 100
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']
        
        # Determine segment characteristics
        # Simple logic to categorize the cluster
        is_recent = recency <= 90  # days
        is_frequent = frequency >= 1.5  # transactions
        is_high_value = monetary >= 5000000  # Rp
        
        # Create segment name and description
        if is_recent and is_frequent and is_high_value:
            segment_name = "Champions"
            segment_desc = "High-value, loyal customers who transacted recently"
        elif is_recent and is_high_value:
            segment_name = "High-Value Recent Customers"
            segment_desc = "High-spending customers who transacted recently but not frequently"
        elif is_frequent and is_high_value:
            segment_name = "Loyal High-Spenders"
            segment_desc = "High-value customers who transact frequently but not recently"
        elif is_recent and is_frequent:
            segment_name = "Loyal Recent Customers"
            segment_desc = "Customers who transact frequently and recently but with lower spending"
        elif is_high_value:
            segment_name = "Big Spenders"
            segment_desc = "High-value customers who don't transact frequently or recently"
        elif is_recent:
            segment_name = "Recently Active"
            segment_desc = "Customers who transacted recently but with low frequency and value"
        elif is_frequent:
            segment_name = "Frequent Buyers"
            segment_desc = "Customers who transact frequently but with low value and not recently"
        else:
            segment_name = "At Risk"
            segment_desc = "Customers who haven't transacted recently and have low frequency and value"
        
        # Add segment information to HTML
        html_content += f"""
            <h3>Segment {cluster}: {segment_name}</h3>
            <p><strong>Description:</strong> {segment_desc}</p>
            <p><strong>Size:</strong> {count:,} customers ({percentage:.1f}% of total)</p>
            <p><strong>Characteristics:</strong></p>
            <ul>
                <li>Average days since last transaction: {recency:.0f}</li>
                <li>Average number of products: {frequency:.1f}</li>
                <li>Average spending: Rp {monetary:,.0f}</li>
            </ul>
            <p><strong>Recommended Approach:</strong> 
                {get_segment_recommendation(segment_name)}
            </p>
        """
    
    # Recommendations section
    html_content += """
        <div class="section">
            <h2>Strategic Recommendations</h2>
            <p>Based on the segmentation analysis, here are strategic recommendations for engaging with each customer segment:</p>
            
            <h3>Overall Recommendations</h3>
            <ul>
                <li>Develop targeted marketing campaigns for each customer segment</li>
                <li>Allocate marketing budget based on segment value and potential</li>
                <li>Create segment-specific messaging and offers</li>
                <li>Measure campaign effectiveness by segment</li>
            </ul>
            
            <h3>Next Steps</h3>
            <ul>
                <li>Develop detailed campaign plans for each segment</li>
                <li>Create communication templates tailored to each segment</li>
                <li>Set up tracking mechanisms to measure campaign performance</li>
                <li>Establish a schedule for refreshing the segmentation analysis</li>
            </ul>
        </div>
    """
    
    return html_content

def generate_campaign_report_content(data, segmented_data, promo_data):
    """
    Generate content for campaign report
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame
        Data hasil segmentasi
    promo_data : pandas.DataFrame
        Data hasil pemetaan promo
    
    Returns:
    --------
    str
        Report content as HTML
    """
    # Campaign statistics
    total_customers = promo_data['Customer_Count'].sum() if 'Customer_Count' in promo_data.columns else 0
    
    # Calculate total budget
    if 'Allocated_Budget' in promo_data.columns and isinstance(promo_data['Allocated_Budget'].iloc[0], str):
        # Convert from string format like "Rp 1,000,000" to numeric
        total_budget = sum([float(b.replace("Rp ", "").replace(",", "")) for b in promo_data['Allocated_Budget']])
    elif 'Allocated_Budget_Num' in promo_data.columns:
        total_budget = promo_data['Allocated_Budget_Num'].sum()
    else:
        total_budget = 0
    
    # Campaign duration
    if 'Start_Date' in promo_data.columns and 'End_Date' in promo_data.columns:
        start_date = promo_data['Start_Date'].iloc[0]
        end_date = promo_data['End_Date'].iloc[0]
        if isinstance(start_date, str):
            # Parse from string if needed
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        duration = (end_date - start_date).days
    else:
        start_date = "N/A"
        end_date = "N/A"
        duration = "N/A"
    
    # HTML content
    html_content = f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report presents a targeted promotional campaign strategy based on customer segmentation.
            The campaign targets {total_customers:,} customers across different segments with a total budget
            of Rp {total_budget:,.0f}.</p>
            
            <p>Each segment has been assigned specific promotional approaches based on their characteristics
            and value to the business. This strategy aims to maximize ROI by allocating resources strategically
            across customer segments.</p>
        </div>
        
        <div class="section">
            <h2>Campaign Overview</h2>
            <p><strong>Campaign Duration:</strong> {start_date} to {end_date} ({duration} days)</p>
            <p><strong>Total Budget:</strong> Rp {total_budget:,.0f}</p>
            <p><strong>Target Audience:</strong> {total_customers:,} customers</p>
            <p><strong>Budget per Customer:</strong> Rp {total_budget/total_customers if total_customers > 0 else 0:,.0f}</p>
            
            <div class="chart-placeholder">
                [Campaign Budget Allocation Chart]
            </div>
        </div>
        
        <div class="section">
            <h2>Segment-Based Campaign Strategy</h2>
            <p>The campaign is tailored to each customer segment based on their RFM characteristics:</p>
    """
    
    # Add segment strategy information
    for i, row in promo_data.iterrows():
        cluster = row['Cluster']
        customer_value = row.get('Customer_Value', 'N/A')
        recency_status = row.get('Recency_Status', 'N/A')
        loyalty_status = row.get('Loyalty_Status', 'N/A')
        customer_count = row.get('Customer_Count', 0)
        promo_type = row.get('Promo_Type', 'N/A')
        promo_desc = row.get('Promo_Description', 'N/A')
        channel = row.get('Channel', 'N/A')
        budget = row.get('Allocated_Budget', 'N/A')
        
        html_content += f"""
            <h3>Segment {cluster}: {customer_value} / {recency_status} Customers</h3>
            <p><strong>Size:</strong> {customer_count:,} customers</p>
            <p><strong>Customer Profile:</strong> {customer_value}, {recency_status}, {loyalty_status}</p>
            <p><strong>Promotion Strategy:</strong> {promo_type}</p>
            <p><strong>Description:</strong> {promo_desc}</p>
            <p><strong>Marketing Channels:</strong> {channel}</p>
            <p><strong>Budget Allocation:</strong> {budget}</p>
        """
    
    # Implementation section
    html_content += """
        <div class="section">
            <h2>Implementation Plan</h2>
            <h3>Timeline</h3>
            <ol>
                <li><strong>Preparation Phase (2 weeks):</strong> Prepare campaign materials, messaging, and channel setup</li>
                <li><strong>Execution Phase (4-6 weeks):</strong> Launch campaigns according to segment priority</li>
                <li><strong>Evaluation Phase (2 weeks):</strong> Measure campaign results and calculate ROI</li>
            </ol>
            
            <h3>Key Performance Indicators (KPIs)</h3>
            <ul>
                <li>Response rate by segment</li>
                <li>Conversion rate by segment</li>
                <li>ROI by segment</li>
                <li>Customer movement between segments post-campaign</li>
                <li>Increase in average transaction value</li>
                <li>Increase in transaction frequency</li>
            </ul>
        </div>
    """
    
    return html_content

def generate_complete_report_content(data, segmented_data, promo_data):
    """
    Generate content for complete analysis report
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame
        Data hasil segmentasi
    promo_data : pandas.DataFrame
        Data hasil pemetaan promo
    
    Returns:
    --------
    str
        Report content as HTML
    """
    # Combine segmentation and campaign report content
    segmentation_content = generate_segmentation_report_content(data, segmented_data)
    campaign_content = generate_campaign_report_content(data, segmented_data, promo_data)
    
    # Executive summary for complete report
    exec_summary = f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This comprehensive report presents both customer segmentation analysis and promotional 
            campaign strategy for SPEKTRA customers. The analysis is based on the RFM (Recency, Frequency, 
            Monetary) framework, which provides insights into customer behavior and value.</p>
            
            <p>Based on the segmentation results, a tailored promotional strategy has been developed 
            for each customer segment, with budget allocation optimized to maximize return on investment.</p>
            
            <p>The report outlines both the analytical findings and strategic recommendations for implementing
            targeted marketing campaigns.</p>
        </div>
    """
    
    # Data overview section
    data_overview = f"""
        <div class="section">
            <h2>Data Overview</h2>
            <p>This analysis is based on a dataset of {data['CUST_NO'].nunique():,} unique customers 
            with transaction history records.</p>
            
            <h3>Key Variables Analyzed</h3>
            <ul>
                <li>Recency: Time since last transaction</li>
                <li>Frequency: Number of products purchased</li>
                <li>Monetary: Total transaction amount</li>
                {"<li>Age: Customer age</li>" if 'Usia' in data.columns else ""}
                {"<li>Gender: Customer gender</li>" if 'CUST_SEX' in data.columns else ""}
                {"<li>Product Categories: Types of products purchased</li>" if 'MPF_CATEGORIES_TAKEN' in data.columns else ""}
            </ul>
            
            <div class="chart-placeholder">
                [Data Distribution Overview Chart]
            </div>
        </div>
    """
    
    # Conclusions and next steps
    conclusions = """
        <div class="section">
            <h2>Conclusions and Next Steps</h2>
            <p>This analysis demonstrates the power of data-driven customer segmentation for targeted marketing.
            By understanding the different customer segments and their unique characteristics, SPEKTRA can
            develop more effective promotional strategies and improve customer engagement.</p>
            
            <h3>Key Takeaways</h3>
            <ul>
                <li>Customer segmentation provides valuable insights into customer behavior and value</li>
                <li>Different segments require different engagement strategies</li>
                <li>Budget allocation should be prioritized based on segment value and potential</li>
                <li>Regular refresh of segmentation will help track customer movement between segments</li>
            </ul>
            
            <h3>Recommended Next Steps</h3>
            <ol>
                <li>Implement the proposed promotional campaigns for each segment</li>
                <li>Set up tracking mechanisms to measure campaign effectiveness</li>
                <li>Create a feedback loop to improve future segmentation and campaign strategies</li>
                <li>Consider expanding the analysis to include more behavioral variables</li>
                <li>Develop a long-term customer engagement strategy based on the segmentation insights</li>
            </ol>
        </div>
    """
    
    # Combine all sections
    html_content = exec_summary + data_overview + segmentation_content + campaign_content + conclusions
    
    return html_content

def get_segment_recommendation(segment_name):
    """
    Get recommendation based on segment name
    
    Parameters:
    -----------
    segment_name : str
        Name of the segment
    
    Returns:
    --------
    str
        Recommendation for the segment
    """
    recommendations = {
        "Champions": "Reward these valuable customers with exclusive benefits, VIP treatment, and premium offers to maintain their loyalty and encourage advocacy.",
        
        "High-Value Recent Customers": "Focus on increasing purchase frequency through personalized product recommendations and loyalty programs.",
        
        "Loyal High-Spenders": "Win back these valuable customers with special reactivation offers and personalized communication.",
        
        "Loyal Recent Customers": "Encourage these customers to increase their spending through upselling and cross-selling strategies.",
        
        "Big Spenders": "Reconnect with these high-value customers through personalized reactivation campaigns highlighting new products or services.",
        
        "Recently Active": "Convert these new or occasional customers into regular ones through engagement programs and targeted offers.",
        
        "Frequent Buyers": "Increase the value of these customers through upselling strategies and premium product recommendations.",
        
        "At Risk": "Implement win-back campaigns with special offers to reactivate these dormant customers."
    }
    
    return recommendations.get(segment_name, "Develop a targeted engagement strategy based on their RFM profile.")
