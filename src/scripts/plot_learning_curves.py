import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve

def plot_learning_curves(best_models_dict, X_train_dict, y_train_dict, targets, model_options=None):
    """
    Create an interactive visualization of learning curves for multiple models and targets.
    
    Parameters:
        best_models_dict (dict): Dictionary of trained models {target: {model_name: model}}
        X_train_dict (dict): Dictionary of training features {target: X_train}
        y_train_dict (dict): Dictionary of training target values {target: y_train}
        targets (list): List of target variable names to analyze
        model_options (list, optional): List of models to include. If None, uses all models in best_models_dict.
        
    Returns:
        plotly.graph_objects.Figure: Interactive learning curve figure
    """
    # If model_options not provided, use all available models for first target
    if model_options is None:
        model_options = list(best_models_dict[targets[0]].keys())
        
    # Create base figure
    fig = go.Figure()
    
    # Professional color palette with rgba values
    colors = {
        'ElasticNet': 'rgba(66, 103, 178, 1)',
        'SVM': 'rgba(66, 183, 42, 1)',
        'GradientBoosting': 'rgba(233, 66, 53, 1)',
        'RandomForest': 'rgba(123, 104, 238, 1)'
    }
    
    # Set default colors for any model not in the predefined palette
    for model in model_options:
        if model not in colors:
            colors[model] = f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 1)'
    
    # Function to create transparent version of color
    def make_transparent(rgba_str, alpha):
        return rgba_str.replace('1)', f'{alpha})')
    
    # Track trace indices for each combination
    trace_indices = {}
    
    # Generate all traces
    for i, target in enumerate(targets):
        trace_indices[target] = {}
        
        X = X_train_dict[target]
        y = y_train_dict[target]
        
        for j, model_name in enumerate(model_options):
            # Skip if model doesn't exist for this target
            if model_name not in best_models_dict[target]:
                continue
                
            # Compute learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                best_models_dict[target][model_name],
                X, y,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Calculate means and standard deviations
            train_mean = -np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = -np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Only show the first target and model by default
            visible = (target == targets[0] and model_name == model_options[0])
            
            # Track traces for this combination
            trace_indices[target][model_name] = []
            
            # Training curve
            idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                name=f"{model_name} (Training)",
                mode='lines+markers',
                line=dict(color=colors[model_name], width=3),
                visible=visible,
                hovertemplate="<b>%{text}</b><br>Size: %{x}<br>MSE: %{y:.4f}<extra></extra>",
                text=[f"{model_name} - Training"]*len(train_sizes)
            ))
            trace_indices[target][model_name].append(idx)
            
            # Training confidence band
            idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                fill='tonexty',
                fillcolor=make_transparent(colors[model_name], 0.2),
                line=dict(width=0),
                showlegend=False,
                visible=visible,
                hoverinfo='skip'
            ))
            trace_indices[target][model_name].append(idx)
            
            # Validation curve
            idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=test_mean,
                name=f"{model_name} (Validation)",
                mode='lines+markers',
                line=dict(color=colors[model_name], width=3, dash='dash'),
                visible=visible,
                hovertemplate="<b>%{text}</b><br>Size: %{x}<br>MSE: %{y:.4f}<extra></extra>",
                text=[f"{model_name} - Validation"]*len(train_sizes)
            ))
            trace_indices[target][model_name].append(idx)
            
            # Validation confidence band
            idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
                fill='tonexty',
                fillcolor=make_transparent(colors[model_name], 0.1),
                line=dict(width=0),
                showlegend=False,
                visible=visible,
                hoverinfo='skip'
            ))
            trace_indices[target][model_name].append(idx)
    
    # Create buttons for target selection
    target_buttons = []
    for target in targets:
        visibility = [False] * len(fig.data)
        
        # Make first model visible for this target
        for i, model_name in enumerate(model_options):
            if model_name in trace_indices[target]:
                if i == 0:  # First model
                    for idx in trace_indices[target][model_name]:
                        visibility[idx] = True
                break
        
        target_buttons.append(
            dict(
                method="update",
                label=target,
                args=[
                    {"visible": visibility},
                    {"title": f"Learning Curves: {target}"}
                ]
            )
        )
    
    # Create buttons for model selection
    model_buttons = []
    for model_name in model_options:
        # Skip models that don't exist for the first target
        if model_name not in trace_indices[targets[0]]:
            continue
            
        visibility = [False] * len(fig.data)
        
        # Show this model for the first target
        for idx in trace_indices[targets[0]][model_name]:
            visibility[idx] = True
        
        model_buttons.append(
            dict(
                method="update",
                label=model_name,
                args=[
                    {"visible": visibility},
                    {"title": f"Learning Curves: {targets[0]} - {model_name}"}
                ]
            )
        )
    
    # Add dropdown menus
    fig.update_layout(
        updatemenus=[
            # Target selector
            dict(
                type="dropdown",
                buttons=target_buttons,
                direction="down",
                active=0,
                x=0.2,
                y=1.15,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(68, 68, 68, 0.8)',
                font=dict(size=12)
            ),
            # Model selector
            dict(
                type="dropdown",
                buttons=model_buttons,
                direction="down",
                active=0,
                x=0.2,
                y=1.10,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(68, 68, 68, 0.8)',
                font=dict(size=12)
            )
        ])
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Learning Curves: {targets[0]} - {model_options[0]}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title={
            'text': 'Training Examples',
            'font': dict(size=14)
        },
        yaxis_title={
            'text': 'Mean Squared Error',
            'font': dict(size=14)
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        margin=dict(t=100, b=80, l=80, r=40),
        template='plotly_white',
        height=700,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_learning_curve_visualization(best_models_dict, X_train_dict, y_train_dict, targets, model_options=None):
    """
    Create and display learning curve visualization for multiple models and targets.
    
    This is a wrapper function that calls plot_learning_curves and adds some additional
    formatting and display options.
    
    Parameters:
        best_models_dict (dict): Dictionary of trained models {target: {model_name: model}}
        X_train_dict (dict): Dictionary of training features {target: X_train}
        y_train_dict (dict): Dictionary of training target values {target: y_train}
        targets (list): List of target variable names to analyze
        model_options (list, optional): List of models to include. If None, uses all models.
        
    Returns:
        plotly.graph_objects.Figure: Interactive learning curve figure
    """
    return plot_learning_curves(
        best_models_dict, 
        X_train_dict, 
        y_train_dict, 
        targets, 
        model_options
    )