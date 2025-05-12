import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Define consistent colors for SHAP visualizations
SHAP_COLORS = {
    'positive': 'rgba(235, 45, 45, 0.8)',  # Red for positive impact
    'negative': 'rgba(35, 70, 200, 0.8)',  # Blue for negative impact
    'neutral': 'rgba(100, 100, 100, 0.7)', # Gray for base values
    'final': 'rgba(35, 150, 35, 0.8)'      # Green for final predictions
}

def analyze_shap_for_model(models_dict, X_test_dict, targets, 
                          model_name='GradientBoosting', 
                          target_name='TotalEnergy(kBtu)',
                          sample_size=50):
    """Calculate SHAP values for a specific model and target"""
    # Validation
    if target_name not in targets:
        print(f"Target {target_name} not found in {targets}")
        return None
    
    if model_name not in models_dict[target_name]:
        print(f"Model {model_name} not found. Available models: {list(models_dict[target_name].keys())}")
        return None
    
    # Get model and data
    model = models_dict[target_name][model_name]
    X_test = X_test_dict[target_name]
    
    # Sample data for speed
    if len(X_test) > sample_size:
        X_test_sample = X_test.sample(sample_size, random_state=42)
    else:
        X_test_sample = X_test
        
    
    # Extract model components
    model_obj = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']
    
    # Preprocess the data
    X_test_processed = preprocessor.transform(X_test_sample)
    
    # Try to get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test_sample.index)
    except:
        feature_names = [f"feature_{i}" for i in range(X_test_processed.shape[1])]
        X_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test_sample.index)
    
    # Create appropriate explainer based on model type
    if model_name in ['GradientBoosting', 'RandomForest']:
        explainer = shap.TreeExplainer(model_obj)
    elif model_name == 'ElasticNet':
        explainer = shap.LinearExplainer(model_obj, X_test_processed)
    else:
        # KernelExplainer - this is much slower
        small_sample = X_test_processed[:min(50, len(X_test_processed))]
        explainer = shap.KernelExplainer(model_obj.predict, small_sample)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_processed)
    
    # Handle case where shap_values is a list (for some models)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]
    
    
    return {
        'explainer': explainer,
        'expected_value': expected_value,
        'shap_values': shap_values,
        'X_processed': X_processed_df,
        'X_original': X_test_sample,
        'feature_names': feature_names,
        'model_name': model_name,
        'target_name': target_name
    }

def plotly_force_plot(shap_result, sample_idx=0):
    """Create an interactive Plotly force plot for a specific sample"""
    if shap_result is None:
        return None
    
    # Get SHAP values for the selected sample
    sample_shap_values = shap_result['shap_values'][sample_idx]
    
    # Sort features by absolute SHAP value
    sorted_idx = np.argsort(-np.abs(sample_shap_values))
    
    # Take top 15 features for clarity
    top_n = 15
    sorted_idx = sorted_idx[:top_n]
    
    # Create lists for force plot
    feature_names = [shap_result['feature_names'][i] for i in sorted_idx]
    feature_values = [shap_result['X_processed'].iloc[sample_idx, i] for i in sorted_idx]
    shap_values = sample_shap_values[sorted_idx]
    
    # Base and final prediction values
    base_value = shap_result['expected_value']
    final_value = base_value + np.sum(sample_shap_values)
    
    # Create a table-like visualization
    fig = go.Figure()
    
    # Add bar for each feature contribution
    for i, (name, value, shap_value) in enumerate(zip(feature_names, feature_values, shap_values)):
        if shap_value > 0:
            color = SHAP_COLORS['positive']  # Red for positive
        else:
            color = SHAP_COLORS['negative']  # Blue for negative
            
        fig.add_trace(go.Bar(
            x=[shap_value],
            y=[i],
            orientation='h',
            marker=dict(color=color),
            width=0.6,
            showlegend=False,
            hovertext=f"{name}: {value:.4g}<br>SHAP value: {shap_value:.4g}",
            hoverinfo='text'
        ))
    
    # Add feature names on the y-axis
    name_text = [f"{name} = {value:.4g}" for name, value in zip(feature_names, feature_values)]
    
    # Update layout
    fig.update_layout(
        title=f"SHAP Force Plot: Sample {sample_idx}<br>{shap_result['model_name']} for {shap_result['target_name']}",
        height=600,
        width=1000,
        template='plotly_white',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(feature_names))),
            ticktext=name_text,
            automargin=True
        ),
        xaxis_title="SHAP Value (impact on model output)",
        margin=dict(l=200)
    )
    
    # Add shapes to show base value and final prediction
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=-1, y1=len(feature_names),
        line=dict(color="gray", width=2)
    )
    
    # Add annotations for base and final values
    fig.add_annotation(
        x=0,
        y=len(feature_names) + 0.5,
        text=f"Base value: {base_value:.4g}",
        showarrow=False,
        font=dict(size=14)
    )
    
    fig.add_annotation(
        x=np.sum(shap_values),
        y=len(feature_names) + 0.5,
        text=f"Final prediction: {final_value:.4g}",
        showarrow=False,
        font=dict(size=14)
    )
    
    return fig

def shap_force_plot(models_dict, X_test_dict, targets,
                  model_name='GradientBoosting',
                  target_name='TotalEnergy(kBtu)',
                  sample_size=50,
                  sample_idx=0):
    """Generate SHAP force plot for a specific sample"""
    # Get SHAP values
    shap_result = analyze_shap_for_model(
        models_dict, X_test_dict, targets,
        model_name, target_name, sample_size
    )
    
    if shap_result is None:
        return None
    
    # Generate force plot
    force_fig = plotly_force_plot(shap_result, sample_idx=sample_idx)
    
    
    return {
        'shap_data': shap_result,
        'force': force_fig
    }