import requests
import random
import time
from datetime import datetime

API_URL = "http://localhost:5000/iot-data"

while True:
    now = datetime.now()

    data = {
        "cars": random.randint(20, 80),
        "bikes": random.randint(30, 120),
        "buses": random.randint(2, 10),
        "trucks": random.randint(5, 20),
        "day": now.weekday(),
        "hour": now.hour
    }

    try:
        response = requests.post(API_URL, json=data)
        print("Sent IoT Data:", data)
    except:
        print("Failed to send data")

    time.sleep(5)  # ⏱ every 5 seconds
