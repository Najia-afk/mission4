"""
Geospatial Visualization Tool for Seattle Building Energy Data

This script provides functions to create interactive maps showing energy consumption,
emissions, and performance metrics for buildings in Seattle. It includes coordinate
transformation to UTM and multiple methods for normalizing data to handle outliers.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
from sklearn.preprocessing import RobustScaler

def normalize_values(series, method='robust', quantile_range=(0.01, 0.99)):
    """
    Normalize values to handle outliers using various methods
    
    Parameters:
    -----------
    series : pd.Series
        The data series to normalize
    method : str
        Normalization method: 'robust', 'minmax', 'percentile', 'log'
    quantile_range : tuple
        For percentile method, the lower and upper percentiles to cap values
        
    Returns:
    --------
    pd.Series: Normalized data series
    """
    if series.isna().all():
        return series
        
    if method == 'robust':
        # Robust scaling is less affected by outliers
        scaler = RobustScaler()
        normalized = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(normalized, index=series.index)
        
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
        
    elif method == 'percentile':
        # Cap values at given percentiles
        lower, upper = series.quantile(quantile_range)
        capped = series.clip(lower, upper)
        # Then apply min-max scaling
        min_val = capped.min()
        max_val = capped.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (capped - min_val) / (max_val - min_val)
        
    elif method == 'log':
        # Log transformation for skewed data
        # Shift to make all values positive if needed
        min_val = series.min()
        if min_val <= 0:
            shifted = series - min_val + 1  # +1 to avoid log(0)
        else:
            shifted = series
        return np.log1p(shifted)  # log(1+x) to handle zeros gracefully
        
    else:
        # Default: return original
        return series

def transform_coordinates(df, lat_col='Latitude', lon_col='Longitude'):
    """
    Transform geographic coordinates from WGS84 to UTM Zone 10N
    with offset from Seattle center
    """
    df_transformed = df.copy()
    
    # Define transformer for WGS84 to UTM Zone 10N
    transformer = Transformer.from_crs("epsg:4326", "epsg:32610", always_xy=True)
    
    # Seattle center coordinates
    seattle_center_lat = 47.620564
    seattle_center_lon = -122.350616
    
    # Transform the Seattle Center coordinates to UTM
    seattle_center_x, seattle_center_y = transformer.transform(seattle_center_lon, seattle_center_lat)
    
    # Transform the coordinates
    df_transformed['X'], df_transformed['Y'] = transformer.transform(
        df[lon_col].astype(float).values, 
        df[lat_col].astype(float).values
    )
    
    # Calculate the offset from Seattle Center
    df_transformed['X'] = df_transformed['X'] - seattle_center_x
    df_transformed['Y'] = df_transformed['Y'] - seattle_center_y
    
    return df_transformed

def create_geo_visualization(df, value_columns=None, normalize_method='percentile', 
                            quantile_range=(0.01, 0.99), lat_col='Latitude', lon_col='Longitude',
                            height=1000, title="Building Energy Data - Seattle"):
    """
    Create an interactive geospatial visualization of building data
    """
    # Transform coordinates
    df_transformed = transform_coordinates(df, lat_col, lon_col)
    
    # Ensure latitude and longitude columns are numeric
    df_transformed[lat_col] = df_transformed[lat_col].astype(float)
    df_transformed[lon_col] = df_transformed[lon_col].astype(float)
    
    # Define default columns to visualize if not provided
    if value_columns is None:
        value_columns = [
            'GHGEmissionsIntensity',
            'SiteEnergyUse(kBtu)',
            'TotalGHGEmissions',
            'ENERGYSTARScore'
        ]
    
    # Normalize each column for visualization
    for col in value_columns:
        if col in df_transformed.columns:
            # Create new column for normalized values
            norm_col = f"{col}_normalized"
            df_transformed[norm_col] = normalize_values(
                df_transformed[col], 
                method=normalize_method,
                quantile_range=quantile_range
            )
    
    # Create the figure
    fig = make_subplots()
    
    # Create plot configurations
    plot_configs = []
    for col in value_columns:
        if col in df_transformed.columns:
            norm_col = f"{col}_normalized"
            plot_configs.append({
                'name': col,
                'lat': df_transformed[lat_col],
                'lon': df_transformed[lon_col],
                'size': 8,
                'color_column': df_transformed[norm_col],
                'colorbar_title': col,
                'hover_text': df_transformed.apply(
                    lambda row: f"{col}: {row[col]:.2f}<br>Lat: {row[lat_col]:.4f}<br>Lon: {row[lon_col]:.4f}", 
                    axis=1
                )
            })
    
    # Add traces to the figure
    for i, config in enumerate(plot_configs):
        fig.add_trace(go.Scattermapbox(
            lat=config['lat'],
            lon=config['lon'],
            mode='markers',
            marker=dict(
                size=config['size'],
                color=config['color_column'],
                colorscale='RdYlGn_r',
                colorbar=dict(title=config['colorbar_title'])
            ),
            name=config['name'],
            hovertext=config['hover_text'],
            hoverinfo='text',
            showlegend=False
        ))
    
    # Update layout for the map
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=47.620564, lon=-122.350616),
            zoom=12
        ),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=18)
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [i == j for j in range(len(plot_configs))]}],
                        label=config['name'],
                        method="update"
                    ) for i, config in enumerate(plot_configs)
                ],
                pad={"r": 8, "t": 8},
                showactive=True,
                x=0.1,
                xanchor="center",
                y=1.05,
                yanchor="top"
            ),
        ],
        height=height,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Set the first trace to be visible by default
    fig.update_traces(visible=False)
    if len(fig.data) > 0:
        fig.data[0].visible = True
    
    return fig, df_transformed
