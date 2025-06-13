#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pytz
import sys
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

# ------------------ SETUP LOGGING ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('/media/home/ssl-fti/ssl_ml/log/anomaly_detection.log', maxBytes=2*1024*1024, backupCount=2),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("/media/home/ssl-fti/ssl_ml/model/knn_model_d.pkl")
    scaler = joblib.load("/media/home/ssl-fti/ssl_ml/scaler/scaler_d.pkl")
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
        logger.debug("Querying InfluxDB")
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
        
        return df
    except Exception as e:
        logger.error(f"Error querying InfluxDB: {e}")
        return pd.DataFrame()

def detect_anomaly(data):
    try:    
        if data.empty or len(data) < lag:
            logger.warning("Insufficient data for anomaly detection")
            return None, None, None, None

        data['lag_24'] = data['y'].shift(7)
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
        data["rolling_mean"] = data["y"].rolling(7).mean()
        data["rolling_std"] = data["y"].rolling(7).std()
        
        tz = pytz.timezone('Asia/Jakarta')
        now = datetime.now(tz)

        day_now = now.day
        # hour_now = now.hour
        minute_now = 0 

        # latest_data = data[(data.index.day == day_now) & (data.index.hour == hour_now) & (data.index.minute == minute_now)]
        latest_data = data[(data.index.day == day_now) & (data.index.minute == minute_now)]
        if latest_data.empty:
            logger.warning("No valid data after feature extraction")
            return None, None, None, None

        data_scaled = scaler.transform(latest_data)

        pred = model.predict(data_scaled)  # 1 = anomali, 0 = normal
        score = model.decision_function(data_scaled)


        return pred[0], score[0], latest_data.index[0], latest_data["y"].values[0]
    except Exception as e:
        logger.error(f"Error during anomaly detection: {e}")
        return None, None, None, None


def run_monitoring():
    try:
        data = get_recent_energy()
        if not data.empty:
            pred, score, timestamp, value = detect_anomaly(data)
            if pred is not None:
                points = (
                    Point("anomaly_prediction")
                    .field("value", float(value))
                    .field("prediction", int(pred))  
                    .field("score", float(score))
                    .time(timestamp.astimezone(pytz.utc), WritePrecision.NS)
                )
                write_api.write(bucket=bucket, org=influx_org, record=points)
                write_api.flush()
                client.close() 
                logger.info("Wrote anomaly prediction results to InfluxDB")

            else:
                logger.warning("Anomaly prediction failed")
        else:
            logger.warning("No data from InfluxDB")
    except Exception as e:
        logger.error(f"Error in run_monitoring: {e}")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    run_monitoring()
    