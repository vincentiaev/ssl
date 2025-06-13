#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import logging
from logging.handlers import RotatingFileHandler
from datetime import timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
import pytz
import sys
from datetime import datetime, timedelta
from influxdb_client.client.write_api import SYNCHRONOUS
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)
from dotenv import load_dotenv
import os

load_dotenv()
# ------------------ KONFIGURASI ------------------
influx_url = os.getenv("INFLUX_URL")
influx_token = os.getenv("INFLUX_TOKEN")
influx_org = os.getenv("INFLUX_ORG")
bucket = os.getenv("INFLUX_BUCKET")


timezone = 'Asia/Jakarta'

lag = 7 
forecast_steps = 31
interval = 1440

# ------------------ SETUP LOGGING ---------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('/media/home/ssl-fti/ssl_ml/log/forecast.log', maxBytes=2*1024*1024, backupCount=2),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("/media/home/ssl-fti/ssl_ml/model/rf_model.pkl")
    scaler_y = joblib.load("/media/home/ssl-fti/ssl_ml/scaler/scaler_y.pkl")
    scaler_X = joblib.load("/media/home/ssl-fti/ssl_ml/scaler/scaler_X.pkl")
    logger.info("Model and scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sys.exit(1)


# ------------------ INFLUXDB CLIENT ------------------
try:
    client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
    query_api = client.query_api()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    logger.info("Connected to InfluxDB")
except Exception as e:
    logger.error(f"Error connecting to InfluxDB: {e}")
    sys.exit(1)

def get_query_range():
    try:
        tz = pytz.timezone('Asia/Jakarta')
        today_local = datetime.now(tz)
        yesterday_local = today_local - timedelta(days=8)
        start_2019 = yesterday_local.replace(year=2019)
        end_2019 = today_local.replace(year=2019)

        start = start_2019.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        end = end_2019.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        return start, end
    except Exception as e:
        logger.error(f"Error generating query range: {e}")
        return None, None



# ------------------ QUERY DATA INFLUX  ------------------
def get_recent_energy():
    try:
        start, end = get_query_range()
        if start is None or end is None:
            return pd.DataFrame()

        query = f'''
        from(bucket: "{bucket}")
        |> range(start: {start}, stop: {end})
        |> filter(fn: (r) => r["_measurement"] == "energy_consumption")
        |> filter(fn: (r) => r["_field"] == "z1_AC4(kW)")
        |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
        |> yield(name: "sum")
        '''
        df = query_api.query_data_frame(query)

        if df.empty:
            logger.warning("No data returned from InfluxDB")
            return pd.DataFrame()

        df = df[["_time", "_value"]].rename(columns={"_time": "time", "_value": "y"})
        df["time"] = pd.to_datetime(df["time"]).dt.tz_convert('Asia/Jakarta')
        df.set_index("time", inplace=True)

        if len(df) < lag:
            logger.warning(f"Insufficient data: {len(df)} rows, need {lag}")
            return pd.DataFrame()

        logger.info(f"Retrieved {len(df)} rows from InfluxDB")
        return df
    except Exception as e:
        logger.error(f"Error querying InfluxDB: {e}")
        return pd.DataFrame()

# ------------------ FORECAST ------------------
def forecast_energy(data):
    try:
        input_lags = data["y"].values[:lag]
        predictions = []
        times = []
        last_time = data.index[lag]

        for step in range(forecast_steps):
            forecast_time = last_time + timedelta(days = step + 1)
            day_of_week = int(forecast_time.dayofweek)
            month = int(forecast_time.month)
            is_weekend = int(day_of_week >= 5)

            features = np.concatenate([input_lags, [day_of_week, month, is_weekend]])
            feature_names = [f"lag_{i+1}" for i in range(lag)] + ["day_of_week", "month", "is_weekend"]
            features_df = pd.DataFrame([features], columns=feature_names)

            features_scaled = scaler_X.transform(features_df)
            pred_scaled = model.predict(features_scaled)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

            predictions.append(pred)
            times.append(forecast_time)
            input_lags = np.append(input_lags, pred)[-lag:]

        forecast_df = pd.DataFrame({"time": times, "forecast_kW": predictions})
        logger.info(f"Generated {len(forecast_df)} predictions")
        return forecast_df
    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        return None
    
# ------------------ MAIN ------------------
def run_forecast():
    try:
        data = get_recent_energy()
        if not data.empty:
            forecast_df = forecast_energy(data)
            if forecast_df is not None:
                # Simpan ke InfluxDB secara batch
                points = [
                    Point("energy_forecast")
                    .field("forecast", float(row["forecast_kW"]))
                    .time(row["time"].astimezone(pytz.utc), WritePrecision.NS)
                    for _, row in forecast_df.iterrows()
                ]
                write_api.write(bucket=bucket, org=influx_org, record=points)
                write_api.flush()
                client.close() 
                logger.info("Predictions written to InfluxDB")

            else:
                logger.warning("Forecasting failed")
        else:
            logger.warning("No data from InfluxDB")
    except Exception as e:
        logger.error(f"Error in run_forecast: {e}")


# ------------------ MAIN ------------------
if __name__ == "__main__":
    logger.info("Starting forecast script")
    run_forecast()
