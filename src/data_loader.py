"""Data loading utilities."""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark import SparkConf
import os


def get_spark_session(app_name="TaxiFarePrediction", master="local[*]"):
    """Get a Spark session."""
    import sys
    import os
    import subprocess
    
    # Get the Python executable path
    python_path = sys.executable
    python_dir = os.path.dirname(python_path)
    
    # Set environment variables BEFORE creating SparkConf
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
    
    # Find SPARK_HOME from pyspark package
    import pyspark
    spark_home = os.path.dirname(pyspark.__file__)
    os.environ["SPARK_HOME"] = spark_home
    
    conf = SparkConf()
    
    # Set Python paths for PySpark workers
    conf.set("spark.python.use.daemon", "false")
    conf.set("spark.python.worker.reuse", "false")
    conf.set("spark.pyspark.python", python_path)
    conf.set("spark.pyspark.driver.python", python_path)
    
    # Java 17+/23 compatibility options - using arrow for memory
    java_options = (
        "-Djava.security.manager=allow "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.pkcs=ALL-UNNAMED "
        "--add-opens=java.base/java.security.cert=ALL-UNNAMED "
        "--add-opens=java.base/java.util.jar=ALL-UNNAMED "
        "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED "
        "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED "
        "-XX:+IgnoreUnrecognizedVMOptions"
    )
    conf.set("spark.driver.extraJavaOptions", java_options)
    conf.set("spark.executor.extraJavaOptions", java_options)
    conf.set("spark.submit.deployMode", "client")
    
    # Reduce memory and use single thread for stability with Java 23
    conf.set("spark.driver.memory", "1g")
    conf.set("spark.executor.memory", "1g")
    conf.set("spark.sql.shuffle.partitions", "2")
    
    # Set environment variables
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
    
    return SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config(conf=conf) \
        .getOrCreate()


def get_schema():
    """Schema for taxi data."""
    return StructType([
        StructField("pickup_datetime", StringType(), True),
        StructField("dropoff_datetime", StringType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("fare_amount", DoubleType(), True)
    ])


def load_data(file_path, spark=None):
    """Load taxi data from CSV."""
    if spark is None:
        spark = get_spark_session()
    
    schema = get_schema()
    df = spark.read.csv(file_path, header=True, schema=schema)
    
    print(f"Loaded {df.count()} records from {file_path}")
    return df


def show_sample_data(df, num_rows=5):
    """Show sample rows."""
    print(f"\nSample ({num_rows} rows):")
    df.show(num_rows)


def print_schema(df):
    """Print schema."""
    print("\nSchema:")
    df.printSchema()


def print_data_info(df):
    """Print info."""
    print(f"\nRecords: {df.count()}, Columns: {len(df.columns)}")

