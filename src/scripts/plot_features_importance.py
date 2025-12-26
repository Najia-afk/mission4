import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_feature_importance(model, preprocessor, X_train):
    """Plot feature importance with proper dimension handling using get_feature_names_out"""
    try:
        # Use the built-in method to get all feature names (including polynomial/one-hot)
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback for older versions or specific configurations
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)
            elif name == 'cat':
                encoder = transformer.named_steps['onehot']
                feature_names.extend(encoder.get_feature_names_out(cols))
    
    # Get importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances.flatten()
    else:
        return None # Skip models without importance metrics
        
    # Ensure dimensions match
    if len(feature_names) != len(importances):
        print(f"DEBUG: {len(feature_names)=}, {len(importances)=}")
        # Try to slice if there's a mismatch (sometimes due to intercept or internal filtering)
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]
        
    # Create DataFrame with aligned data
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Aggregate and sort
    importance_df = (importance_df.groupby('feature')['importance']
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index())
    
    # Add cumulative importance
    importance_df['cumulative'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
    
    # Filter to top 95% contributors
    return importance_df[importance_df['cumulative'] <= 0.95]

# Create interactive visualization
def create_importance_visualization(feature_importance_data, targets, models):
    """Create interactive feature importance plot with target selection"""
    
    # Filter models to only those that actually have data in feature_importance_data
    available_models = {}
    for model_name in models.keys():
        # Check if this model has data for ALL requested targets
        has_all_targets = True
        for target in targets:
            if target not in feature_importance_data or model_name not in feature_importance_data[target]:
                has_all_targets = False
                break
        if has_all_targets:
            available_models[model_name] = models[model_name]
    
    if not available_models:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data available for the selected models/targets",
                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))
        return fig

    # Create base figure
    fig = go.Figure()
    
    # Add traces for each combination
    initial_target = targets[0]
    initial_model = list(available_models.keys())[0]
    
    for target in targets:
        for model_name in available_models.keys():
            data = feature_importance_data[target][model_name]
            if data is not None and not data.empty:
                fig.add_trace(go.Bar(
                    x=data['feature'],
                    y=data['importance'],
                    name=f"{target} - {model_name}",
                    visible=(target == initial_target and model_name == initial_model)
                ))
    
    # Create model dropdown buttons
    model_buttons = [{
        'label': model,
        'method': 'update',
        'args': [
            {'visible': [
                t == initial_target and m == model 
                for t in targets 
                for m in available_models.keys()
            ]},
            {'title': f"Feature Importance: {initial_target} ({model})"}
        ]
    } for model in available_models.keys()]
    
    # Create target dropdown buttons
    target_buttons = [{
        'label': target,
        'method': 'update',
        'args': [
            {'visible': [
                t == target and m == initial_model 
                for t in targets 
                for m in available_models.keys()
            ]},
            {'title': f"Feature Importance: {target} ({initial_model})"}
        ]
    } for target in targets]
    
    # Update layout with both dropdowns
    fig.update_layout(
        updatemenus=[
            # Model selector
            {
                'buttons': model_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.5,
                'xanchor': 'center',
                'y': 1.27,
                'yanchor': 'middle'
            },
            # Target selector
            {
                'buttons': target_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.7,
                'xanchor': 'center',
                'y': 1.27,
                'yanchor': 'middle'
            }
        ],
        title=f"Feature Importance: {initial_target} ({initial_model})",
        xaxis_title="Features",
        yaxis_title="Importance",
        showlegend=False
    )
    
    return fig