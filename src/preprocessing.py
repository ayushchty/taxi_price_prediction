"""Data preprocessing utilities."""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, unix_timestamp


def filter_valid_data(df):
    """Remove bad records."""
    df_filtered = df.filter(
        (col('fare_amount') > 0) &
        (col('fare_amount').isNotNull()) &
        (col('pickup_longitude').isNotNull()) &
        (col('pickup_latitude').isNotNull()) &
        (col('dropoff_longitude').isNotNull()) &
        (col('dropoff_latitude').isNotNull())
    )
    
    bad = df.count() - df_filtered.count()
    if bad > 0:
        print(f"Filtered {bad} bad records")
    return df_filtered


def parse_datetime(df, datetime_column='pickup_datetime'):
    """Parse datetime strings."""
    df = df.withColumn(
        datetime_column, 
        unix_timestamp(datetime_column, 'yyyy-MM-dd HH:mm:ss').cast('timestamp')
    )
    return df


def remove_duplicates(df):
    """Drop duplicates."""
    initial = df.count()
    df = df.distinct()
    removed = initial - df.count()
    if removed > 0:
        print(f"Removed {removed} duplicates")
    return df


def handle_missing_values(df):
    """Drop rows with nulls."""
    initial = df.count()
    df = df.dropna()
    removed = initial - df.count()
    if removed > 0:
        print(f"Dropped {removed} null rows")
    return df


def preprocess(df):
    """Run preprocessing."""
    print("\n=== Preprocessing ===")
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = filter_valid_data(df)
    df = parse_datetime(df)
    print("=== Done ===\n")
    return df

