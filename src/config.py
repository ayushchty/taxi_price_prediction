# Project config

# Spark Configuration
SPARK_APP_NAME = "TaxiFarePrediction"
SPARK_MASTER = "local[*]"

# Data Paths
DATA_PATH = "data/sample.csv"
OUTPUT_PATH = "output/"

# Model Configuration
TRAIN_TEST_SPLIT = [0.8, 0.2]
RANDOM_SEED = 123

# Feature Columns
FEATURE_COLUMNS = ['hour', 'weekday', 'distance', 'passenger_count']
LABEL_COLUMN = 'fare_amount'
OUTPUT_COLUMN = 'features'

# Data Schema
SCHEMA = {
    'pickup_datetime': 'string',
    'dropoff_datetime': 'string',
    'passenger_count': 'int',
    'pickup_longitude': 'double',
    'pickup_latitude': 'double',
    'dropoff_longitude': 'double',
    'dropoff_latitude': 'double',
    'fare_amount': 'double'
}

# Data Validation Rules
MIN_FARE_AMOUNT = 0
MAX_FARE_AMOUNT = 500
MIN_LONGITUDE = -180
MAX_LONGITUDE = 180
MIN_LATITUDE = -90
MAX_LATITUDE = 90

