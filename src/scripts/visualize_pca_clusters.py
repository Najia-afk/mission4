from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer

def perform_pca_analysis(df, n_components=None, target_cols=None):
    """
    Perform PCA on the dataset and create visualizations.
    
    Parameters:
    -----------
    df : DataFrame
        The transformed dataframe
    n_components : int, optional
        Number of components to keep (defaults to min(n_samples, n_features))
    target_cols : list, optional
        Target columns to use for coloring the scatter plot
        
    Returns:
    --------
    fig_var : plotly figure
        Explained variance plot
    fig_scatter : plotly figure
        Principal components scatter plot
    fig_loadings : plotly figure
        Feature loadings plot
    pca_results : dict
        Dictionary with PCA results and transformed data
    """
    # Select only numerical columns
    num_df = df.select_dtypes(include=['number'])
    
    # Handle any remaining missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(num_df)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create dataframe with PCA results
    pca_df = pd.DataFrame(
        X_pca, 
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
        index=df.index
    )
    
    # Add target columns if provided
    if target_cols:
        for col in target_cols:
            if col in df.columns:
                pca_df[col] = df[col].values
    
    # Create explained variance plot
    explained_var = pca.explained_variance_ratio_ * 100
    cum_explained_var = np.cumsum(explained_var)
    
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_var))],
        y=explained_var,
        name='Individual Explained Variance',
        marker_color='rgb(55, 83, 109)'
    ))
    fig_var.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(cum_explained_var))],
        y=cum_explained_var,
        name='Cumulative Explained Variance',
        marker_color='rgb(26, 118, 255)',
        mode='lines+markers'
    ))
    fig_var.update_layout(
        title='Explained Variance by Principal Component',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance (%)',
        yaxis=dict(range=[0, 105]),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        template='plotly_white'
    )
    
    # Create scatter plot of first two principal components
    if target_cols and target_cols[0] in pca_df.columns:
        fig_scatter = px.scatter(
            pca_df, x='PC1', y='PC2',
            color=target_cols[0],
            hover_data=[col for col in pca_df.columns if col.startswith('PC')] + target_cols,
            title=f'PCA Scatter Plot Colored by {target_cols[0]}',
            template='plotly_white',
            color_continuous_scale='viridis'
        )
    else:
        fig_scatter = px.scatter(
            pca_df, x='PC1', y='PC2',
            hover_data=[col for col in pca_df.columns if col.startswith('PC')],
            title='PCA Scatter Plot',
            template='plotly_white'
        )
    
    # Create loadings plot
    loadings = pca.components_.T
    feature_names = num_df.columns.tolist()
    
    loadings_df = pd.DataFrame(
        loadings[:, :2],  # Just first two components
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    # Sort features by magnitude of PC1 loading
    loadings_df['PC1_abs'] = loadings_df['PC1'].abs()
    loadings_df = loadings_df.sort_values('PC1_abs', ascending=False).drop('PC1_abs', axis=1)
    
    # Select top 20 features with highest loading magnitudes
    top_features = loadings_df.iloc[:20].index.tolist()
    
    fig_loadings = px.scatter(
        loadings_df.loc[top_features],
        x='PC1', y='PC2',
        text=loadings_df.loc[top_features].index,
        title='Feature Loadings on First Two Principal Components',
        template='plotly_white'
    )
    
    fig_loadings.update_traces(
        textposition='top center',
        marker=dict(size=10, color='rgba(55, 83, 109, 0.7)',
                   line=dict(width=1, color='DarkSlateGrey'))
    )
    
    # Add lines from origin to each point
    for i, feature in enumerate(top_features):
        x = loadings_df.loc[feature, 'PC1']
        y = loadings_df.loc[feature, 'PC2']
        fig_loadings.add_shape(
            type='line',
            x0=0, y0=0, x1=x, y1=y,
            line=dict(color='rgba(55, 83, 109, 0.3)', width=1)
        )
    
    # Add a circle to represent correlation of 1
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    fig_loadings.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1),
            showlegend=False
        )
    )
    
    fig_loadings.update_layout(
        xaxis=dict(
            title='PC1',
            range=[-1.1, 1.1],
            zeroline=True, zerolinewidth=1, zerolinecolor='black'
        ),
        yaxis=dict(
            title='PC2', 
            range=[-1.1, 1.1],
            zeroline=True, zerolinewidth=1, zerolinecolor='black'
        )
    )
    
    # Package results
    pca_results = {
        'pca_model': pca,
        'explained_variance': explained_var,
        'cumulative_variance': cum_explained_var,
        'transformed_data': pca_df,
        'loadings': loadings,
        'feature_names': feature_names
    }
    
    return fig_var, fig_scatter, fig_loadings, pca_results

