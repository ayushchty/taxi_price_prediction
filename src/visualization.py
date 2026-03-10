"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    """Set plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)


def plot_prediction_scatter(predictions_pd):
    """Scatter plot of actual vs predicted."""
    set_style()
    plt.figure(figsize=(10, 8))
    
    plt.scatter(predictions_pd['fare_amount'], 
                predictions_pd['prediction'], 
                alpha=0.5, 
                edgecolors='none')
    
    max_val = max(predictions_pd['fare_amount'].max(), predictions_pd['prediction'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
    
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual ($)')
    plt.ylabel('Predicted ($)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(predictions_pd):
    """Residual analysis."""
    set_style()
    
    predictions_pd = predictions_pd.copy()
    predictions_pd['residuals'] = predictions_pd['fare_amount'] - predictions_pd['prediction']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(predictions_pd['prediction'], predictions_pd['residuals'], alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted ($)')
    axes[0].set_ylabel('Residual ($)')
    axes[0].set_title('Residuals vs Predicted')
    
    axes[1].hist(predictions_pd['residuals'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residual ($)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()


def create_model_summary_plot(metrics, feature_importance=None):
    """Summary dashboard."""
    set_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Metrics
    names = list(metrics.keys())
    values = list(metrics.values())
    axes[0, 0].bar(names, values, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Model Metrics')
    axes[0, 0].set_ylabel('Value')
    
    # Feature importance
    if feature_importance:
        features = [f[0] for f in feature_importance]
        importance = [f[1] for f in feature_importance]
        axes[0, 1].barh(features, importance, color='steelblue', alpha=0.7)
        axes[0, 1].set_title('Feature Importance')
    else:
        axes[0, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        axes[0, 1].set_title('Feature Importance')
    
    # RMSE
    rmse_val = metrics.get('RMSE', metrics.get('rmse', 'N/A'))
    rmse_str = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else str(rmse_val)
    axes[1, 0].text(0.5, 0.5, f"RMSE: {rmse_str}", 
                    ha='center', va='center', fontsize=24, fontweight='bold')
    axes[1, 0].set_title('RMSE')
    axes[1, 0].axis('off')
    
    # R2
    r2_val = metrics.get('R2', metrics.get('r2', 'N/A'))
    r2_str = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val)
    axes[1, 1].text(0.5, 0.5, f"R²: {r2_str}", 
                    ha='center', va='center', fontsize=24, fontweight='bold')
    axes[1, 1].set_title('R² Score')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

