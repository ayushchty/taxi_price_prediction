"""ML models for taxi fare prediction."""

from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def train_random_forest(df, features_col='features', label_col='fare_amount', 
                         num_trees=20, max_depth=10, seed=123):
    """Train Random Forest."""
    rf = RandomForestRegressor(
        featuresCol=features_col,
        labelCol=label_col,
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=seed
    )
    
    print(f"\n=== Training RF (trees={num_trees}, depth={max_depth}) ===")
    model = rf.fit(df)
    print("=== Done ===\n")
    return model


def train_gbt(df, features_col='features', label_col='fare_amount', 
              max_iter=20, max_depth=5, seed=123):
    """Train GBT."""
    gbt = GBTRegressor(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=max_iter,
        maxDepth=max_depth,
        seed=seed
    )
    
    print(f"\n=== Training GBT (iter={max_iter}, depth={max_depth}) ===")
    model = gbt.fit(df)
    print("=== Done ===\n")
    return model


def train_linear_regression(df, features_col='features', label_col='fare_amount',
                             max_iter=100, reg_param=0.0):
    """Train Linear Regression."""
    lr = LinearRegression(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=max_iter,
        regParam=reg_param
    )
    
    print(f"\n=== Training LR (iter={max_iter}) ===")
    model = lr.fit(df)
    print("=== Done ===\n")
    return model


def split_data(df, train_ratio=0.8, test_ratio=0.2, seed=123):
    """Split data."""
    train_data, test_data = df.randomSplit([train_ratio, test_ratio], seed=seed)
    
    print(f"\n=== Split: {train_data.count()} train / {test_data.count()} test ===")
    return train_data, test_data


def make_predictions(model, df):
    """Make predictions."""
    return model.transform(df)


def evaluate_model(predictions, label_col='fare_amount', prediction_col='prediction'):
    """Calculate metrics."""
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col
    )
    
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    
    print(f"\n=== Metrics ===")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print("===============\n")
    
    return {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2': r2}


def train_and_evaluate(df, model_type='rf', feature_col='features', label_col='fare_amount',
                       train_ratio=0.8, seed=123):
    """Train and evaluate."""
    train_data, test_data = split_data(df, train_ratio, 1-train_ratio, seed)
    
    if model_type.lower() == 'rf':
        model = train_random_forest(train_data, feature_col, label_col)
    elif model_type.lower() == 'gbt':
        model = train_gbt(train_data, feature_col, label_col)
    elif model_type.lower() == 'lr':
        model = train_linear_regression(train_data, feature_col, label_col)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    predictions = make_predictions(model, test_data)
    metrics = evaluate_model(predictions, label_col, 'prediction')
    
    return model, predictions, metrics


def get_feature_importance(model):
    """Get feature importance."""
    try:
        importance = model.featureImportances.toArray()
        features = ['hour', 'weekday', 'distance', 'passenger_count']
        
        feature_importance = list(zip(features, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n=== Feature Importance ===")
        for f, imp in feature_importance:
            print(f"  {f}: {imp:.4f}")
        print("==========================\n")
        return feature_importance
    except Exception as e:
        print(f"Error: {e}")
        return None

