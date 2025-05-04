import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway

def identify_numerical_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()

def identify_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def analyze_variable_relationships(df):
    """Comprehensive analysis of relationships between variables of different types"""
    # Get column types
    numerical_cols = identify_numerical_columns(df)
    categorical_cols = identify_categorical_columns(df)
    
    # 1. Numerical-Numerical Relationships: Correlation Matrix
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr(method='spearman')
        
        # Plot correlation heatmap
        fig_num = px.imshow(
            corr_matrix, 
            text_auto=True, 
            color_continuous_scale='RdBu_r',
            title="Numerical Variable Correlations (Spearman)"
        )
        fig_num.update_layout(height=900)
        fig_num.show()
    
    # 2. Categorical-Categorical Relationships: Cramer's V
    if len(categorical_cols) > 1:
        # Fill missing values for categorical columns
        df_cat = df[categorical_cols].fillna('missing')
        cramers_v_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
        
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 == col2:
                    cramers_v_matrix.loc[col1, col2] = 1.0
                else:
                    confusion_matrix = pd.crosstab(df_cat[col1], df_cat[col2])
                    chi2 = chi2_contingency(confusion_matrix)[0]
                    n = confusion_matrix.values.sum()
                    phi2 = chi2 / n
                    r, k = confusion_matrix.shape
                    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                    rcorr = r - ((r-1)**2)/(n-1)
                    kcorr = k - ((k-1)**2)/(n-1)
                    cramers_v_matrix.loc[col1, col2] = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        
        # Plot Cramer's V heatmap
        mask = np.triu(np.ones_like(cramers_v_matrix, dtype=bool))
        cramers_v_matrix_masked = cramers_v_matrix.mask(mask)
        
        fig_cat = px.imshow(
            cramers_v_matrix_masked, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale='Blues',
            title="Categorical Variable Associations (Cramer's V)"
        )
        fig_cat.update_layout( height=900)
        fig_cat.show()
    
    # 3. Mixed Relationships: Categorical vs. Numerical
    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        # Create DataFrame to store association strengths
        mixed_assoc = pd.DataFrame(index=categorical_cols, columns=numerical_cols)
        
        # Calculate association using ANOVA F-value
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                # Skip if too many missing values
                if df[cat_col].isna().sum() > 0.5 * len(df) or df[num_col].isna().sum() > 0.5 * len(df):
                    mixed_assoc.loc[cat_col, num_col] = np.nan
                    continue
                    
                # Group data by categorical variable and perform ANOVA
                groups = []
                for category in df[cat_col].dropna().unique():
                    group_data = df[df[cat_col] == category][num_col].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                
                if len(groups) > 1:  # Need at least two groups for ANOVA
                    try:
                        f_value, p_value = f_oneway(*groups)
                        if np.isnan(f_value):
                            mixed_assoc.loc[cat_col, num_col] = 0
                        else:
                            # Convert to a normalized association measure (0 to 1)
                            # Using the formula: η² = (f * (num_groups - 1)) / (f * (num_groups - 1) + (total_samples - num_groups))
                            num_groups = len(groups)
                            total_samples = sum(len(group) for group in groups)
                            eta_squared = (f_value * (num_groups - 1)) / (f_value * (num_groups - 1) + (total_samples - num_groups))
                            mixed_assoc.loc[cat_col, num_col] = eta_squared
                    except:
                        mixed_assoc.loc[cat_col, num_col] = np.nan
                else:
                    mixed_assoc.loc[cat_col, num_col] = np.nan
        
        # Plot mixed association heatmap
        fig_mixed = px.imshow(
            mixed_assoc, 
            text_auto=True, 
            color_continuous_scale='Viridis',
            title="Categorical-Numerical Variable Association (η² from ANOVA)"
        )
        fig_mixed.update_layout(height=900)
        fig_mixed.show()
        
    return {
        'numerical_correlations': corr_matrix if len(numerical_cols) > 1 else None,
        'categorical_associations': cramers_v_matrix if len(categorical_cols) > 1 else None,
        'mixed_associations': mixed_assoc if (len(categorical_cols) > 0 and len(numerical_cols) > 0) else None
    }