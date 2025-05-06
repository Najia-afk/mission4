import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

def visualize_model_metrics_comparison(best_models_dict, X_test_dict, y_test_dict, targets, colors=None):
    """
    Create a visualization comparing performance metrics for multiple models and target variables.

    """
   # Improved color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Create figures for metrics visualization
    fig_metrics = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Mean Squared Error", "Root Mean Squared Error", 
                    "Mean Absolute Error", "R² Score"),
        vertical_spacing=0.2,
        horizontal_spacing=0.15
    )

    # Calculate and store metrics
    metrics_data = {target: {
        'model_names': [],
        'mse': [],
        'rmse': [],
        'mae': [], 
        'r2': []
    } for target in targets}

    # Calculate metrics with improved visualization data
    for target in targets:
        X_test = X_test_dict[target]
        y_test = y_test_dict[target]
        
        for idx, (model_name, model) in enumerate(best_models_dict[target].items()):
            y_pred = model.predict(X_test)
            
            metrics_data[target]['model_names'].append(model_name)
            metrics_data[target]['mse'].append(mean_squared_error(y_test, y_pred))
            metrics_data[target]['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics_data[target]['mae'].append(mean_absolute_error(y_test, y_pred))
            metrics_data[target]['r2'].append(r2_score(y_test, y_pred))

    # Improved bar plots with distinct colors and better spacing
    for i, target in enumerate(targets):
        for metric, row, col in [('mse', 1, 1), ('rmse', 1, 2), ('mae', 2, 1), ('r2', 2, 2)]:
            fig_metrics.add_trace(
                go.Bar(
                    name=f"{target}",
                    x=metrics_data[target]['model_names'],
                    y=metrics_data[target][metric],
                    marker_color=[colors[j + i*len(metrics_data[target]['model_names'])] 
                                for j in range(len(metrics_data[target]['model_names']))],
                    text=np.round(metrics_data[target][metric], 3),
                    textposition='outside',
                    showlegend=(row == 1 and col == 1),
                    legendgroup=target,  # Link traces with same target
                    visible='legendonly' if i > 0 else True  # Show first target by default
                ),
                row=row, col=col
            )

    # Improved metrics layout
    fig_metrics.update_layout(
        height=1000,  # Increased height for better readability
        title={
            'text': "Comparison of Model Performance Metrics",
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        barmode='group',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        showlegend=True,
        font=dict(size=12)
    )

    # Update axes with better formatting
    for i in range(1, 3):
        for j in range(1, 3):
            fig_metrics.update_xaxes(row=i, col=j, tickangle=45)

    fig_metrics.update_xaxes(title_text="Models", row=2, col=1, title_font=dict(size=14))
    fig_metrics.update_xaxes(title_text="Models", row=2, col=2, title_font=dict(size=14))
    fig_metrics.update_yaxes(title_text="MSE", row=1, col=1, title_font=dict(size=14))
    fig_metrics.update_yaxes(title_text="RMSE", row=1, col=2, title_font=dict(size=14))
    fig_metrics.update_yaxes(title_text="MAE", row=2, col=1, title_font=dict(size=14))
    fig_metrics.update_yaxes(title_text="R² Score", row=2, col=2, title_font=dict(size=14))

    fig_metrics.show()