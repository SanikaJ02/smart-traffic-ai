import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv("Traffic.csv")
data.head()

data.tail()

# data.shape

data.describe()

data.info()

data.isnull().sum()

# Convert Time to datetime
data["Time"] = pd.to_datetime(data["Time"])

# Extract hour
data["Hour"] = data["Time"].dt.hour

# Encode day of week
day_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

data['Day of the week'] = data['Day of the week'].map(day_mapping)

# Drop unused columns
data.drop(columns=["Time", "Date","Total"], inplace=True)

data.head()

data.head()

traffic_mapping = {
    'low': 0,
    'normal': 1,
    'high': 2,
    'heavy':3
}

data['Traffic_Situation_Encoded'] = data['Traffic Situation'].map(traffic_mapping)

data.head()

sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="Blues")
plt.show()

import matplotlib.pyplot as plt

plt.figure()

plt.boxplot(
    [
        data[data['Traffic_Situation_Encoded'] == 0]['CarCount'],
        data[data['Traffic_Situation_Encoded'] == 1]['CarCount'],
        data[data['Traffic_Situation_Encoded'] == 2]['CarCount'],
        data[data['Traffic_Situation_Encoded'] == 3]['CarCount']
    ],
    labels=['Low', 'normal', 'High','heavy']
)

plt.xlabel('Traffic Situation')
plt.ylabel('Car Count')
plt.title('Car Count Distribution by Traffic Situation')
plt.show()

X_traffic = data[
    ["CarCount", "BikeCount", "BusCount", "TruckCount", "Hour", "Day of the week"]
]

y_traffic = data["Traffic Situation"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_traffic, y_traffic, test_size=0.2, random_state=42
)

traffic_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

traffic_model.fit(X_tr, y_tr)

traffic_pred = traffic_model.predict(X_te)

from sklearn.metrics import classification_report

print("Traffic Situation Classification Report:\n")
print(classification_report(y_te, traffic_pred))

print("Traffic Prediction Accuracy:",
      accuracy_score(y_te, traffic_pred))

data["CO2_Label"] = (
    data["BikeCount"] * 40 +
    data["CarCount"] * 120 +
    data["BusCount"] * 900 +
    data["TruckCount"] * 800
)

data[["CO2_Label"]].head()

X_co2 = data[
    ["CarCount", "BikeCount", "BusCount", "TruckCount", "Hour", "Day of the week"]
]

y_co2 = data["CO2_Label"]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_co2, y_co2, test_size=0.2, random_state=42
)

co2_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)

co2_model.fit(X_tr2, y_tr2)

co2_pred = co2_model.predict(X_te2)

print("CO₂ Prediction MAE:",
      mean_absolute_error(y_te2, co2_pred))
print("R² Score:", r2_score(y_te2, co2_pred))

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(y_te2, co2_pred)
plt.xlabel("Actual CO2 Emission")
plt.ylabel("Predicted CO2 Emission")
plt.title("Actual vs Predicted CO2 Emission (Random Forest Regressor)")
plt.show()

mean_co2 = data.groupby('Traffic_Situation_Encoded')['CO2_Label'].mean()

plt.figure()
plt.plot(mean_co2.index, mean_co2.values, marker='o')
plt.xlabel('Traffic Situation')
plt.ylabel('Average CO2 Emission')
plt.title('Average CO2 Emission per Traffic Level')
plt.show()

def signal_timing(traffic, co2):
    traffic = traffic.lower()

    if traffic == "heavy" or (co2 > 45000):
        return "Green Signal: 150 sec"
    elif traffic == "high" or (30000 < co2 <= 45000):
        return "Green Signal: 90 sec"
    elif traffic == "normal" or (20000 < co2 <= 30000):
        return "Green Signal: 60 sec"
    else:
        return "Green Signal: 30 sec"

print("🚦 Signal Optimization Results\n")

for i in range(5):
    print(
        f"Traffic: {traffic_pred[i]} | "
        f"Predicted CO₂: {int(co2_pred[i])} → "
        f"{signal_timing(traffic_pred[i], co2_pred[i])}"
    )

def predict_signal(car, bike, bus, truck, day, hour):

    # Create input dataframe
    input_df = pd.DataFrame([{
        "CarCount": car,
        "BikeCount": bike,
        "BusCount": bus,
        "TruckCount": truck,
        "Hour": hour,
        "Day of the week": day
    }])

    # Predict traffic
    traffic = traffic_model.predict(input_df)[0]

    # Predict CO2
    co2 = co2_model.predict(input_df)[0]

    # Signal optimization
    signal = signal_timing(traffic, co2)

    return traffic, int(co2), signal
import streamlit as st
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Traffic & CO₂ Optimization",
    page_icon="🚦",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🚦 AI-based Traffic & CO₂ Aware Signal Optimization</h1>
    <p style='text-align: center; color: gray;'>
    Predict traffic condition, estimate CO₂ emissions, and optimize signal timing using ML
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- MODE SELECTION ----------------
mode = st.radio(
    "📡 Select Data Input Mode",
    ["Manual Input", "IoT Real-Time Mode"],
    horizontal=True
)

st.subheader("📥 Traffic Details")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

if mode == "Manual Input":

    with col1:
        car_count = st.number_input("🚗 Car Count", min_value=0, step=1)
        bike_count = st.number_input("🏍️ Bike Count", min_value=0, step=1)
        bus_count = st.number_input("🚌 Bus Count", min_value=0, step=1)

    with col2:
        truck_count = st.number_input("🚛 Truck Count", min_value=0, step=1)
        day = st.selectbox(
            "📅 Day of Week",
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
        )
        hour = st.slider("⏰ Hour of Day", 0, 23, 12)

else:
    # -------- IOT MODE --------
    try:
        iot_data = requests.get("http://localhost:5000/get-latest").json()

        car_count = iot_data["cars"]
        bike_count = iot_data["bikes"]
        bus_count = iot_data["buses"]
        truck_count = iot_data["trucks"]
        day = iot_data["day"]
        hour = iot_data["hour"]

        with col1:
            st.number_input("🚗 Car Count", value=car_count, disabled=True)
            st.number_input("🏍️ Bike Count", value=bike_count, disabled=True)
            st.number_input("🚌 Bus Count", value=bus_count, disabled=True)

        with col2:
            st.number_input("🚛 Truck Count", value=truck_count, disabled=True)
            st.selectbox(
                "📅 Day of Week",
                options=[0,1,2,3,4,5,6],
                index=day,
                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
                disabled=True
            )
            st.slider("⏰ Hour of Day", 0, 23, hour, disabled=True)

        st.info("📡 Live IoT sensor data received")

    except:
        st.error("❌ IoT server not running. Start Flask API & simulator.")
        st.stop()

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Traffic & Signal Timing", use_container_width=True):

    traffic, emission, signal_time = predict_signal(
        car_count,
        bike_count,
        bus_count,
        truck_count,
        day,
        hour
    )

    st.success("✅ Prediction Successful")

    st.subheader("📊 Results")

    col3, col4, col5 = st.columns(3)

    col3.metric("🚦 Traffic Situation", traffic)
    col4.metric("🌫️ CO₂ Emission (g/min)", emission)

    with col5:
        st.markdown(
            f"""
            <div style="
                background-color:#e8f5e9;
                padding:18px;
                border-radius:10px;
                text-align:center;
                font-size:18px;
                font-weight:bold;
                color:#2e7d32;">
                🟢 Green Signal Time<br>
                {signal_time}
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

st.markdown(
    "<p style='text-align:center; color:gray;'>Academic Prototype – AI-based Smart Traffic Management 🚘</p>",
    unsafe_allow_html=True
)


