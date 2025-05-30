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
    Fungsi untuk menampilkan halaman ekspor dan dokumentasi (tanpa promo mapping)
    """
    st.markdown('<p class="section-title">Export & Documentation</p>', unsafe_allow_html=True)
    
    # Check for data
    if st.session_state.data is None:
        st.warning("Please upload and preprocess data first.")
        return
    
    # Get data from session state
    data = st.session_state.data
    segmented_data = st.session_state.segmented_data if st.session_state.segmentation_completed else None
    
    # Show export options
    st.markdown("### Export Options")
    
    tab1, tab2, tab3 = st.tabs(["Export Data", "Export Report", "Export Charts"])
    
    with tab1:
        show_data_export(data, segmented_data)
    
    with tab2:
        show_report_export(data, segmented_data)
    
    with tab3:
        show_chart_export(data, segmented_data)
    
    # Show documentation
    st.markdown("### Documentation")
    show_documentation()

def show_data_export(data, segmented_data):
    """
    Tampilkan opsi ekspor data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
    """
    st.markdown("#### Export Data to CSV/Excel")
    
    # Data selection
    export_options = ["Processed Customer Data"]
    if segmented_data is not None:
        export_options.append("Segmentation Results")
        export_options.append("All Data")
    
    export_option = st.radio("Select data to export:", export_options)
    
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
    else:  # All Data
        # Create a list of DataFrames to export
        export_dfs = {"processed_data": data}
        
        if segmented_data is not None:
            export_dfs["segmented_data"] = segmented_data
    
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
            
            st.download_button(
                label="Download All Data (Excel)",
                data=output.getvalue(),
                file_name="spektra_customer_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

def show_report_export(data, segmented_data):
    """
    Tampilkan opsi ekspor laporan
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
    """
    st.markdown("#### Generate Analytical Report")
    
    # Report options (simplified)
    report_options = ["Customer Analysis Report"]
    if segmented_data is not None:
        report_options.append("Customer Segmentation Report")
    
    report_type = st.selectbox("Select report type:", report_options)
    
    # Report details
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name", "FIFGROUP")
        report_title = st.text_input("Report Title", f"SPEKTRA {report_type}")
    
    with col2:
        author = st.text_input("Author/Team", "Data Science Team")
        date = st.date_input("Report Date", datetime.datetime.now())
    
    # Check prerequisites for report generation
    if report_type == "Customer Segmentation Report" and segmented_data is None:
        st.warning("Segmentation data required for this report. Please complete the segmentation first.")
        can_generate = False
    else:
        can_generate = True
    
    if can_generate and st.button("Generate Report"):
        with st.spinner("Generating report..."):
            # Build report content
            report_content = generate_report_content(
                report_type, data, segmented_data,
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

def show_chart_export(data, segmented_data):
    """
    Tampilkan opsi ekspor chart
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
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
        chart_options.append("Invitation Status by Segment")
    
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
    Tampilkan dokumentasi (simplified version)
    """
    st.markdown("#### SPEKTRA Customer Segmentation App Documentation")
    
    with st.expander("About This Application"):
        st.markdown("""
        The SPEKTRA Customer Segmentation App is a comprehensive analytics tool designed to help
        marketing teams segment customers based on RFM (Recency, Frequency, Monetary) analysis.
        
        Key features include:
        - Data preprocessing and cleaning
        - Exploratory data analysis
        - RFM (Recency, Frequency, Monetary) analysis
        - K-means clustering for customer segmentation
        - Customer invitation recommendations
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
           - Choose between Optimal Method or Standard RFM K-means
           - Perform customer segmentation and get invitation recommendations
        
        4. **Dashboard**
           - View an integrated dashboard of all analysis
           - Get key insights about your customer segments
           - See invitation summary and recommendations
        
        5. **Export & Documentation**
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
    
    with st.expander("Segmentation Methods"):
        st.markdown("""
        ### Available Segmentation Methods
        
        **1. Optimal Method (Research-Based)**
        - Based on research in financial services industry
        - Uses 4 segments: Potential Loyalists, Responsive Customers, Occasional Buyers, Hibernating Customers
        - Enhanced features: Log transformation, repeat customer flag, prime age segment
        - Dynamic naming based on cluster characteristics
        - Automatic invitation recommendations
        
        **2. Standard RFM K-Means**
        - Traditional RFM segmentation
        - Customizable number of clusters (2-10)
        - Optional z-score normalization
        - Manual interpretation of cluster characteristics
        
        ### Invitation Logic
        - Customers in top-performing segments receive invitation recommendations
        - Based on combined RFM scores and segment characteristics
        - Helps prioritize marketing efforts on most valuable customers
        """)

def generate_report_content(report_type, data, segmented_data, 
                           company_name, report_title, author, date):
    """
    Generate report content based on type (simplified)
    
    Parameters:
    -----------
    report_type : str
        Type of report to generate
    data : pandas.DataFrame
        Data pelanggan
    segmented_data : pandas.DataFrame or None
        Data hasil segmentasi
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
            .metric-box {{
                background-color: #f0f8ff;
                border: 1px solid #003366;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
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
    
    # Add content based on report type
    if report_type == "Customer Segmentation Report":
        html_content += generate_segmentation_report_content(data, segmented_data)
    else:  # Customer Analysis Report
        html_content += generate_analysis_report_content(data)
    
    # Add footer and close HTML
    html_content += """
        <div class="footer">
            <p>Generated by SPEKTRA Customer Segmentation App</p>
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
    
    # Invitation statistics
    invitation_col = None
    if 'Layak_Diundang_optimal' in segmented_data.columns:
        invitation_col = 'Layak_Diundang_optimal'
    elif 'Invitation_Status' in segmented_data.columns:
        invitation_col = 'Invitation_Status'
    
    invited_count = 0
    if invitation_col:
        invited_count = len(segmented_data[segmented_data[invitation_col].str.contains('✅', na=False)])
    
    # HTML content
    html_content = f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report presents the results of customer segmentation analysis based on RFM methodology
            (Recency, Frequency, Monetary). {total_customers:,} unique customers were analyzed and grouped
            into {n_clusters} distinct segments.</p>
            
            <p>Based on the segmentation analysis, {invited_count:,} customers ({invited_count/total_customers*100:.1f}% of total)
            are recommended for targeted marketing campaigns.</p>
            
            <div class="metric-box">
                <h3>Key Metrics</h3>
                <ul>
                    <li>Total Customers Analyzed: {total_customers:,}</li>
                    <li>Number of Segments: {n_clusters}</li>
                    <li>Customers Recommended for Invitation: {invited_count:,} ({invited_count/total_customers*100:.1f}%)</li>
                    <li>Average Customer Age: {avg_age:.1f} years</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Segmentation Methodology</h2>
            <p>The segmentation was performed using the RFM (Recency, Frequency, Monetary) framework combined with K-means clustering:</p>
            <ul>
                <li><strong>Recency:</strong> How recently a customer made a transaction (days since last purchase)</li>
                <li><strong>Frequency:</strong> How often a customer makes transactions (number of products purchased)</li>
                <li><strong>Monetary:</strong> How much money a customer spends (total transaction value)</li>
            </ul>
            <p>K-means clustering with {n_clusters} clusters was used to group similar customers based on their normalized RFM values.</p>
        </div>
        
        <div class="section">
            <h2>Segment Profiles</h2>
            <div class="chart-placeholder">
                [Cluster Distribution Visualization]
            </div>
    """
    
    # Add cluster information if segmentation was completed
    if segmented_data is not None:
        cluster_stats = segmented_data.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CUST_NO': 'count'
        }).reset_index()
        
        cluster_stats.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
        
        # Add segment descriptions
        for _, row in cluster_stats.iterrows():
            cluster = row['Cluster']
            count = row['CUST_NO']
            percentage = count / total_customers * 100
            recency = row['Avg_Recency']
            frequency = row['Avg_Frequency']
            monetary = row['Avg_Monetary']
            
            # Get segment name if available
            segment_name = f"Cluster {cluster}"
            if 'Segmentasi_optimal' in segmented_data.columns:
                segment_name = segmented_data[segmented_data['Cluster'] == cluster]['Segmentasi_optimal'].iloc[0]
            
            # Get invitation status
            invite_status = "Not Available"
            if invitation_col:
                invite_status = segmented_data[segmented_data['Cluster'] == cluster][invitation_col].iloc[0]
            
            # Add segment information to HTML
            html_content += f"""
                <h3>{segment_name}</h3>
                <div class="metric-box">
                    <p><strong>Size:</strong> {count:,} customers ({percentage:.1f}% of total)</p>
                    <p><strong>Invitation Status:</strong> {invite_status}</p>
                    <p><strong>Characteristics:</strong></p>
                    <ul>
                        <li>Average days since last transaction: {recency:.0f}</li>
                        <li>Average number of products: {frequency:.1f}</li>
                        <li>Average spending: Rp {monetary:,.0f}</li>
                    </ul>
                </div>
            """
    
    # Add recommendations
    html_content += """
        </div>
        
        <div class="section">
            <h2>Strategic Recommendations</h2>
            <h3>Immediate Actions</h3>
            <ul>
                <li>Focus marketing campaigns on customers with invitation recommendations</li>
                <li>Develop segment-specific messaging and offers</li>
                <li>Prioritize high-value segments for premium campaigns</li>
                <li>Create reactivation campaigns for inactive segments</li>
            </ul>
            
            <h3>Long-term Strategy</h3>
            <ul>
                <li>Monitor customer movement between segments over time</li>
                <li>Refresh segmentation analysis quarterly</li>
                <li>Measure campaign effectiveness by segment</li>
                <li>Expand analysis to include additional behavioral variables</li>
            </ul>
        </div>
    """
    
    return html_content

def generate_analysis_report_content(data):
    """
    Generate content for general customer analysis report
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data pelanggan
    
    Returns:
    --------
    str
        Report content as HTML
    """
    # Basic statistics
    total_customers = data['CUST_NO'].nunique()
    avg_age = data['Usia'].mean() if 'Usia' in data.columns else "N/A"
    avg_transaction = data['TOTAL_AMOUNT_MPF'].mean() if 'TOTAL_AMOUNT_MPF' in data.columns else "N/A"
    
    html_content = f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report provides a comprehensive analysis of {total_customers:,} customers 
            based on their transaction history and demographic information.</p>
            
            <div class="metric-box">
                <h3>Key Customer Metrics</h3>
                <ul>
                    <li>Total Customers: {total_customers:,}</li>
                    <li>Average Customer Age: {avg_age:.1f} years</li>
                    <li>Average Transaction Value: Rp {avg_transaction:,.0f}</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Customer Demographics</h2>
            <div class="chart-placeholder">
                [Age and Gender Distribution Charts]
            </div>
            <p>The customer base shows diverse demographic characteristics across different age groups and segments.</p>
        </div>
        
        <div class="section">
            <h2>Transaction Patterns</h2>
            <div class="chart-placeholder">
                [Transaction Analysis Charts]
            </div>
            <p>Analysis of transaction patterns reveals insights into customer purchasing behavior and product preferences.</p>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <h3>Next Steps</h3>
            <ul>
                <li>Proceed with customer segmentation analysis to identify distinct customer groups</li>
                <li>Develop targeted marketing strategies based on customer characteristics</li>
                <li>Monitor key performance indicators regularly</li>
                <li>Consider expanding data collection for deeper insights</li>
            </ul>
        </div>
    """
    
    return html_content
