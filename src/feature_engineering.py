"""Feature engineering for taxi fare prediction."""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, hour, dayofweek, dayofmonth, month, year, lit, sqrt, sin, cos, atan2, radians
from pyspark.ml.feature import VectorAssembler
import math


def calculate_distance_spark(df, pickup_lon='pickup_longitude', pickup_lat='pickup_latitude', 
                              dropoff_lon='dropoff_longitude', dropoff_lat='dropoff_latitude'):
    """
    Calculate haversine distance using native Spark SQL functions (no Python UDFs).
    This avoids Python worker connection issues on Windows.
    """
    # Convert to radians
    lat1 = radians(col(pickup_lat))
    lat2 = radians(col(dropoff_lat))
    lon1 = radians(col(pickup_lon))
    lon2 = radians(col(dropoff_lon))
    
    # Haversine formula using Spark SQL functions
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Earth's radius in km
    R = lit(6371.0)
    distance = R * c
    
    df = df.withColumn('distance', distance)
    return df


def extract_time_features(df, datetime_column='pickup_datetime'):
    """Extract time features."""
    df = df.withColumn('hour', hour(datetime_column))
    df = df.withColumn('weekday', dayofweek(datetime_column))
    df = df.withColumn('day', dayofmonth(datetime_column))
    df = df.withColumn('month', month(datetime_column))
    df = df.withColumn('year', year(datetime_column))
    return df


def create_feature_vector(df, feature_columns, output_column='features'):
    """Combine features into vector."""
    assembler = VectorAssembler(inputCols=feature_columns, outputCol=output_column)
    return assembler.transform(df)


def engineer_features(df, feature_columns=['hour', 'weekday', 'distance', 'passenger_count']):
    """Run feature engineering."""
    print("\n=== Feature Engineering ===")
    df = extract_time_features(df)
    df = calculate_distance_spark(df)  # Use native Spark functions instead of Python UDF
    df = create_feature_vector(df, feature_columns)
    print(f"Features: {feature_columns}")
    print("=== Done ===\n")
    return df


def get_feature_columns():
    """Return feature columns."""
    return ['hour', 'weekday', 'distance', 'passenger_count']

