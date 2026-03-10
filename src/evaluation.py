"""Model evaluation utilities."""

from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd


def calculate_rmse(predictions, label_col='fare_amount', prediction_col='prediction'):
    """Calculate RMSE."""
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='rmse'
    )
    return evaluator.evaluate(predictions)


def calculate_mse(predictions, label_col='fare_amount', prediction_col='prediction'):
    """Calculate MSE."""
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='mse'
    )
    return evaluator.evaluate(predictions)


def calculate_mae(predictions, label_col='fare_amount', prediction_col='prediction'):
    """Calculate MAE."""
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='mae'
    )
    return evaluator.evaluate(predictions)


def calculate_r2(predictions, label_col='fare_amount', prediction_col='prediction'):
    """Calculate R-squared."""
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='r2'
    )
    return evaluator.evaluate(predictions)


def evaluate_all_metrics(predictions, label_col='fare_amount', prediction_col='prediction'):
    """Calculate all metrics."""
    metrics = {
        'RMSE': calculate_rmse(predictions, label_col, prediction_col),
        'MSE': calculate_mse(predictions, label_col, prediction_col),
        'MAE': calculate_mae(predictions, label_col, prediction_col),
        'R2': calculate_r2(predictions, label_col, prediction_col)
    }
    
    print("\n=== Metrics ===")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")
    print("===============\n")
    return metrics


def save_predictions(predictions, output_path):
    """Save predictions to CSV."""
    pandas_df = predictions.toPandas()
    pandas_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def save_metrics(metrics, output_path):
    """Save metrics to file."""
    with open(output_path, 'w') as f:
        f.write("=== Metrics ===\n\n")
        for name, val in metrics.items():
            f.write(f"{name}: {val:.4f}\n")
    print(f"Saved: {output_path}")

