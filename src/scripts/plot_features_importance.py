import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_feature_importance(model, preprocessor, X_train):
    """Plot feature importance with proper dimension handling"""
    # Get feature names first
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            # Get one-hot encoded feature names
            encoder = transformer.named_steps['onehot']
            encoder.fit(X_train[cols])
            feature_names.extend(encoder.get_feature_names_out(cols))
    
    # Get importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances.flatten()
    else:
        raise ValueError("Model does not provide feature importances")
        
    # Ensure dimensions match
    if len(feature_names) != len(importances):
        raise ValueError(f"Feature names ({len(feature_names)}) and importances ({len(importances)}) dimensions mismatch")
        
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
    
    # Create base figure
    fig = go.Figure()
    
    # Add traces for each combination
    initial_target = targets[0]
    initial_model = list(models.keys())[0]
    
    for target in targets:
        for model_name in models.keys():
            data = feature_importance_data[target][model_name]
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
                for m in models.keys()
            ]},
            {'title': f"Feature Importance: {initial_target} ({model})"}
        ]
    } for model in models.keys()]
    
    # Create target dropdown buttons
    target_buttons = [{
        'label': target,
        'method': 'update',
        'args': [
            {'visible': [
                t == target and m == initial_model 
                for t in targets 
                for m in models.keys()
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