"""
Taxi Fare Prediction Pipeline using Spark.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_spark_session, load_data, show_sample_data, print_schema, print_data_info
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features, get_feature_columns
from src.model import train_and_evaluate, get_feature_importance
from src.evaluation import save_predictions, save_metrics
from src.visualization import create_model_summary_plot, plot_prediction_scatter, plot_residuals
import src.config as config


def run_pipeline(data_path='data/sample.csv', model_type='rf'):
    """Run the full pipeline."""
    print("=" * 50)
    print("TAXI FARE PREDICTION (Spark)")
    print("=" * 50)
    
    # Start Spark
    print("\nStarting Spark...")
    spark = get_spark_session(config.SPARK_APP_NAME, config.SPARK_MASTER)
    print("Done!")
    
    # Load
    print("\nLoading data...")
    df = load_data(data_path, spark)
    show_sample_data(df, 5)
    print_schema(df)
    print_data_info(df)
    
    # Preprocess
    print("\nPreprocessing...")
    df = preprocess(df)
    show_sample_data(df, 5)
    print_data_info(df)
    
    # Features
    print("\nEngineering features...")
    feature_columns = get_feature_columns()
    df = engineer_features(df, feature_columns)
    show_sample_data(df, 5)
    
    # Train
    print("\nTraining model...")
    model, predictions, metrics = train_and_evaluate(
        df, 
        model_type=model_type,
        feature_col=config.OUTPUT_COLUMN,
        label_col=config.LABEL_COLUMN,
        train_ratio=config.TRAIN_TEST_SPLIT[0],
        seed=config.RANDOM_SEED
    )
    
    feature_importance = get_feature_importance(model)
    
    # Visualize
    print("\nVisualizing...")
    predictions_pd = predictions.select(config.LABEL_COLUMN, 'prediction').toPandas()
    
    create_model_summary_plot(metrics, feature_importance)
    plot_prediction_scatter(predictions_pd)
    plot_residuals(predictions_pd)
    
    # Save
    print("\nSaving results...")
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    save_predictions(predictions, os.path.join(config.OUTPUT_PATH, 'predictions.csv'))
    save_metrics(metrics, os.path.join(config.OUTPUT_PATH, 'metrics.txt'))
    
    # Cleanup
    spark.stop()
    
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    
    return {
        'model': model,
        'predictions': predictions,
        'metrics': metrics,
        'feature_importance': feature_importance
    }


def main():
    """Main entry point."""
    data_path = config.DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"\nData not found at {data_path}")
        print("Use main_pandas.py to generate data first!")
        return
    
    model_type = 'rf'
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    results = run_pipeline(data_path, model_type)
    
    print("\n" + "=" * 50)
    print("FINAL METRICS")
    print("=" * 50)
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

