import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_cv_results_comparison(cv_results_dict, targets):
    """
    Creates two types of plots for each target:
    1. Performance plot: shows mean squared error by model and hyperparameters
    2. R² score plot: shows R² by model and hyperparameters
    """
    
    # Create figures for MSE and R²
    fig_mse = make_subplots(
        rows=1, cols=len(targets),
        subplot_titles=[f"{target} - Mean Squared Error" for target in targets],
        horizontal_spacing=0.15
    )
    
    fig_r2 = make_subplots(
        rows=1, cols=len(targets),
        subplot_titles=[f"{target} - R² Score" for target in targets],
        horizontal_spacing=0.15
    )

    # Professional color palette
    colors = {
        'ElasticNet': '#4267B2',
        'SVM': '#42B72A',
        'GradientBoosting': '#E94235',
        'RandomForest': '#7B68EE'
    }

    # Plot for each target
    for i, target in enumerate(targets, 1):
        target_cv_results = cv_results_dict[target]
        
        for model_name in colors.keys():
            model_results = target_cv_results[target_cv_results['model'] == model_name]
            
            if not model_results.empty:
                # Get scores and parameters
                mse_means = -model_results['mean_test_neg_mean_squared_error']
                r2_means = model_results['mean_test_r2']
                params = model_results['params'].apply(lambda x: {k.split('__')[1]: v for k, v in x.items()})
                
                # Create parameter labels
                param_labels = [f"Config {j+1}" for j in range(len(model_results))]
                
                # Add MSE trace
                fig_mse.add_trace(
                    go.Scatter(
                        x=param_labels,
                        y=mse_means,
                        name=model_name,
                        mode='lines+markers',
                        marker=dict(
                            size=10,
                            symbol='circle',
                            line=dict(width=2, color='white')
                        ),
                        line=dict(color=colors[model_name], width=2),
                        hovertemplate=(
                            f"<b>{model_name}</b><br>" +
                            "Parameters: %{text}<br>" +
                            "MSE: %{y:.4f}<extra></extra>"
                        ),
                        text=[str(p) for p in params]
                    ),
                    row=1, col=i
                )
                
                # Add R² trace
                fig_r2.add_trace(
                    go.Scatter(
                        x=param_labels,
                        y=r2_means,
                        name=model_name,
                        mode='lines+markers',
                        marker=dict(
                            size=10,
                            symbol='circle',
                            line=dict(width=2, color='white')
                        ),
                        line=dict(color=colors[model_name], width=2),
                        hovertemplate=(
                            f"<b>{model_name}</b><br>" +
                            "Parameters: %{text}<br>" +
                            "R²: %{y:.4f}<extra></extra>"
                        ),
                        text=[str(p) for p in params]
                    ),
                    row=1, col=i
                )

    # Update layouts
    for fig, title, yaxis in [(fig_mse, "MSE for Different Model Configurations", "Mean Squared Error"),
                             (fig_r2, "R² Scores for Different Model Configurations", "R² Score")]:
        fig.update_layout(
            height=600,
            width=1200,
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                font=dict(size=24)
            ),
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=100, b=100)
        )
        
        # Update axes for all subplots
        for j in range(1, len(targets) + 1):
            fig.update_xaxes(
                row=1, col=j,
                title="Hyperparameter Configurations",
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            )
            fig.update_yaxes(
                row=1, col=j,
                title=yaxis,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            )

    return fig_mse, fig_r2