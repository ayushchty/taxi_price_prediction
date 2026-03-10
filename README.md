# Taxi Fare Prediction Project

An end-to-end taxi fare prediction system using Apache Spark and Machine Learning.

## Project Overview

This project predicts taxi fare amounts based on pickup/dropoff locations, time of day, and passenger count using a Random Forest Regressor.

### Features
- **hour**: Hour of the day (extracted from pickup_datetime)
- **weekday**: Day of the week 
- **distance**: Distance between pickup and dropoff (Haversine formula)
- **passenger_count**: Number of passengers

### Tech Stack
- **Apache Spark** (PySpark) - Distributed data processing
- **Random Forest Regressor** - ML model
- **Seaborn/Matplotlib** - Visualization

## Project Structure

```
taxi_price_prediction/
├── README.md
├── requirements.txt
├── main.py                 # Entry point
├── .gitignore
├── data/
│   └── sample.csv         # Sample taxi data
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessing.py   # Data cleaning
│   ├── feature_engineering.py  # Feature extraction
│   ├── model.py           # ML model training
│   ├── evaluation.py      # Model evaluation
│   └── visualization.py   # Plotting results
└── output/
    ├── predictions.csv    # Model predictions
    ├── metrics.txt        # Evaluation metrics
    └── predictions.png    # Visualization plots
```

## Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Spark Setup (PySpark handles this automatically)

PySpark will automatically download Spark when first run. To verify:

```bash
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.getOrCreate(); print(spark.version)"
```

If you need manual setup, download from: https://spark.apache.org/downloads.html

```bash
# Windows - Extract and set:
set SPARK_HOME=C:\path\to\spark-3.4.1
set PATH=%PATH%;%SPARK_HOME\bin

# Linux/Mac
export SPARK_HOME=/path/to/spark-3.4.1
export PATH=$PATH:$SPARK_HOME/bin
```

## Running the Project

### Run from main.py

```bash
python main.py
```

This will:
- Automatically create sample data if not present
- Run the complete pipeline
- Save predictions and metrics

### Command Line Options

```bash
python main.py rf    # Random Forest (default)
python main.py gbt   # Gradient Boosted Trees
python main.py lr    # Linear Regression
```

## Sample Data Format

| Column | Type | Description |
|--------|------|-------------|
| pickup_datetime | string | Pickup time (YYYY-MM-DD HH:MM:SS) |
| dropoff_datetime | string | Dropoff time (YYYY-MM-DD HH:MM:SS) |
| passenger_count | int | Number of passengers |
| pickup_longitude | float | Pickup longitude |
| pickup_latitude | float | Pickup latitude |
| dropoff_longitude | float | Dropoff longitude |
| dropoff_latitude | float | Dropoff latitude |
| fare_amount | float | Fare in dollars |

## Model Performance

- **Algorithm**: Random Forest Regressor
- **Evaluation Metric**: RMSE (Root Mean Square Error)
- **Train/Test Split**: 80/20

### Results

| Metric | Value |
|--------|-------|
| RMSE | 3.5188 |
| MSE | 12.3821 |
| MAE | 2.5486 |
| R² | 0.9244 |

### Feature Importance

| Feature | Importance |
|---------|------------|
| distance | 0.9258 |
| hour | 0.0373 |
| weekday | 0.0223 |
| passenger_count | 0.0146 |

The model achieves **92.44% R² score**, indicating excellent prediction accuracy. Distance is by far the most important feature (92.58%), followed by time-based features.



