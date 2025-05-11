import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings

def calculate_residuals(best_models_dict, X_test_dict, y_test_dict, transformers_dict=None):
    """Calculate residuals for all models and targets in original scale if transformers provided."""
    residual_data = {}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, 
                              message="X does not have valid feature names")
        
        for target, models in best_models_dict.items():
            residual_data[target] = {}
            for model_name, model in models.items():
                # Get predictions in transformed space
                y_pred_log = model.predict(X_test_dict[target])
                y_test_log = y_test_dict[target]
                
                # Calculate residuals (either in original or log space)
                if transformers_dict and target in transformers_dict:
                    # Convert to original space first
                    transformer = transformers_dict[target]
                    y_pred_orig = transformer.inverse_transform(y_pred_log)
                    y_test_orig = transformer.inverse_transform(y_test_log)
                    residuals = y_test_orig - y_pred_orig  # ORIGINAL UNITS
                else:
                    # Keep in log space
                    residuals = y_test_log - y_pred_log
                
                residual_data[target][model_name] = residuals
    
    return residual_data

def create_interactive_residual_analysis(best_models_dict, X_test_dict, y_test_dict, targets=None, transformers_dict=None):
    """
    Create an interactive dashboard for residual analysis.
    
    Parameters:
        best_models_dict (dict): Dictionary of best models {target: {model_name: model}}
        X_test_dict (dict): Dictionary of test features {target: X_test}
        y_test_dict (dict): Dictionary of test target values {target: y_test}
        targets (list): List of target variable names to analyze (defaults to all targets in best_models_dict)
        
    Returns:
        plotly.graph_objects.Figure: Interactive residual analysis figure
    """
    # Get targets if not provided
    if targets is None:
        targets = list(best_models_dict.keys())
    
    # Get model names (assuming same models for all targets)
    model_options = list(best_models_dict[targets[0]].keys())
    
    # Calculate residuals for all models and targets
    residual_data = calculate_residuals(best_models_dict, X_test_dict, y_test_dict, transformers_dict)
    
    # Create tabs for different visualization types
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Residuals vs Predicted", "Residual Distribution", 
                         "Q-Q Plot", "Residual Statistics"],
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Get initial data to display
    initial_target = targets[0]
    initial_model = model_options[0]
    
    # Track traces for each combination to fix visibility toggling
    trace_indices = {}
    
    # Data for all combinations (will toggle visibility)
    for target in targets:
        trace_indices[target] = {}
        for model_name in model_options:
            trace_indices[target][model_name] = []
            
            # Get prediction and residuals (with warning suppressed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, 
                                      message="X does not have valid feature names")
                y_pred = best_models_dict[target][model_name].predict(X_test_dict[target])
            
            residuals = residual_data[target][model_name]
            
            # Is this the initial view?
            visible = (target == initial_target and model_name == initial_model)
            
            # 1. Residuals vs Predicted scatter plot
            idx = len(fig.data)

            # Get x-values in correct scale
            if transformers_dict and target in transformers_dict:
                # Use original scale predictions for x-axis
                transformer = transformers_dict[target]
                x_values = transformer.inverse_transform(y_pred)
            else:
                # Keep in log scale
                x_values = y_pred
                
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=residuals,
                    mode='markers',
                    name=f"{model_name} - {target}",
                    visible=visible,
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        line=dict(width=1)
                    )
                ),
                row=1, col=1
            )
            trace_indices[target][model_name].append(idx)
            
            # Add reference line at y=0
            idx = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=[min(y_pred), max(y_pred)],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False,
                    visible=visible
                ),
                row=1, col=1
            )
            trace_indices[target][model_name].append(idx)
            
            # 2. Histogram of residuals
            idx = len(fig.data)
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name=f"Distribution",
                    visible=visible,
                    marker_color='rgba(56, 108, 176, 0.7)'
                ),
                row=1, col=2
            )
            trace_indices[target][model_name].append(idx)
            
            # 3. Q-Q Plot
            idx = len(fig.data)
            qq = stats.probplot(residuals, dist="norm", plot=None)
            fig.add_trace(
                go.Scatter(
                    x=qq[0][0],
                    y=qq[0][1],
                    mode='markers',
                    name='Residuals',
                    visible=visible
                ),
                row=2, col=1
            )
            trace_indices[target][model_name].append(idx)
            
            idx = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=qq[0][0],
                    y=qq[0][0],
                    mode='lines',
                    name='Reference Line',
                    visible=visible,
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            trace_indices[target][model_name].append(idx)
            
            # 4. Residual statistics table
            idx = len(fig.data)
            stats_data = [
                ['Mean', f"{np.mean(residuals):.4f}"],
                ['Median', f"{np.median(residuals):.4f}"],
                ['Std Dev', f"{np.std(residuals):.4f}"],
                ['Min', f"{np.min(residuals):.4f}"],
                ['Max', f"{np.max(residuals):.4f}"],
                ['Skewness', f"{stats.skew(residuals):.4f}"],
                ['Kurtosis', f"{stats.kurtosis(residuals):.4f}"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Statistic', 'Value'],
                               fill_color='paleturquoise',
                               align='left'),
                    cells=dict(values=list(zip(*stats_data)),
                              fill_color='lavender',
                              align='left'),
                    visible=visible
                ),
                row=2, col=2
            )
            trace_indices[target][model_name].append(idx)
    
    # Create dropdown buttons for model selection
    model_buttons = []
    for model_name in model_options:
        visibility = [False] * len(fig.data)
        
        # Set visibility based on actual trace indices
        if initial_target in trace_indices and model_name in trace_indices[initial_target]:
            for idx in trace_indices[initial_target][model_name]:
                visibility[idx] = True
                
        model_buttons.append(
            dict(
                label=model_name,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Residual Analysis: {initial_target} with {model_name}"}]
            )
        )
    
    # Create dropdown buttons for target selection
    target_buttons = []
    for target in targets:
        visibility = [False] * len(fig.data)
        
        # Set visibility based on actual trace indices
        if target in trace_indices and initial_model in trace_indices[target]:
            for idx in trace_indices[target][initial_model]:
                visibility[idx] = True
                
        target_buttons.append(
            dict(
                label=target,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Residual Analysis: {target} with {initial_model}"}]
            )
        )
    
    # Add dropdown menus to layout
    fig.update_layout(
        updatemenus=[
            # Target selector
            dict(
                active=0,
                buttons=target_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.5)",
            ),
            # Model selector
            dict(
                active=0,
                buttons=model_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.2,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.5)",
            ),
        ]
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title={
            'text': f"Residual Analysis: {initial_target} with {initial_model}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18}
        },
        template="plotly_white",
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    
    return fig