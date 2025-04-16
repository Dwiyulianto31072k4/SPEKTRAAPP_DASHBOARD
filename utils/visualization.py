import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_distribution_plot(data, column, title=None, log_transform=False, color='#003366'):
    """
    Create distribution plot for a numeric column
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to plot
    column : str
        Column name to plot
    title : str, optional
        Plot title
    log_transform : bool, default=False
        Whether to apply log transformation
    color : str, default='#003366'
        Color for the plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if log_transform and (data[column] > 0).all():
        fig = px.histogram(
            data, 
            x=np.log1p(data[column]), 
            title=title or f"Log Distribution of {column}",
            nbins=50,
            color_discrete_sequence=[color]
        )
        fig.update_layout(xaxis_title=f"Log({column})")
    else:
        fig = px.histogram(
            data, 
            x=column, 
            title=title or f"Distribution of {column}",
            nbins=50,
            color_discrete_sequence=[color]
        )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_categorical_bar(data, column, top_n=10, title=None, color_scale=px.colors.sequential.Blues):
    """
    Create bar chart for a categorical column
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to plot
    column : str
        Column name to plot
    top_n : int, default=10
        Number of top categories to show
    title : str, optional
        Plot title
    color_scale : list, default=px.colors.sequential.Blues
        Color scale for the plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    value_counts = data[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']
    
    fig = px.bar(
        value_counts.head(top_n), 
        x=column, 
        y='Count',
        title=title or f"Top {top_n} values of {column}",
        color='Count',
        color_continuous_scale=color_scale
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_pie_chart(data, column, title=None, hole=0.4, colors=px.colors.sequential.Blues):
    """
    Create pie chart for a categorical column
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to plot
    column : str
        Column name to plot
    title : str, optional
        Plot title
    hole : float, default=0.4
        Size of the hole (0-1)
    colors : list, default=px.colors.sequential.Blues
        Color sequence for the plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    counts = data[column].value_counts().reset_index()
    counts.columns = [column, 'Count']
    
    fig = px.pie(
        counts, 
        values='Count', 
        names=column,
        title=title or f"{column} Distribution",
        hole=hole,
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_scatter_plot(data, x, y, color=None, size=None, title=None, opacity=0.7, 
                      color_scale=px.colors.sequential.Blues):
    """
    Create scatter plot
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to plot
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    color : str, optional
        Column name for color
    size : str, optional
        Column name for size
    title : str, optional
        Plot title
    opacity : float, default=0.7
        Opacity of points (0-1)
    color_scale : list, default=px.colors.sequential.Blues
        Color scale for the plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = px.scatter(
        data, 
        x=x, 
        y=y,
        color=color,
        size=size,
        title=title or f"{y} vs {x}",
        opacity=opacity,
        color_continuous_scale=color_scale
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_rfm_3d_plot(data, title="3D RFM Visualization"):
    """
    Create 3D scatter plot for RFM analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with Recency, Frequency, Monetary, and Cluster columns
    title : str, default="3D RFM Visualization"
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = px.scatter_3d(
        data,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        title=title,
        opacity=0.7
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Recency',
            yaxis_title='Frequency',
            zaxis_title='Monetary'
        ),
        height=600
    )
    
    return fig

def create_radar_chart(data, columns, group_col, title="Radar Chart"):
    """
    Create radar chart
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to plot
    columns : list
        List of column names to plot
    group_col : str
        Column name for grouping
    title : str, default="Radar Chart"
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        values = group_data[columns].mean().tolist()
        values.append(values[0])  # Close the loop
        
        labels = columns + [columns[0]]  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=f"{group_col}: {group}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        title=title,
        height=500
    )
    
    return fig

def create_cluster_bar_chart(data, group_col, metrics, title="Cluster Metrics"):
    """
    Create bar chart for cluster metrics
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster metrics
    group_col : str
        Column name for grouping (e.g., 'Cluster')
    metrics : list
        List of metric column names to plot
    title : str, default="Cluster Metrics"
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            x=data[group_col],
            y=data[metric],
            name=metric
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=group_col,
        yaxis_title="Value",
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig
