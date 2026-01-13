from flask import Flask, request, jsonify

app = Flask(__name__)

# Store latest IoT data (acts like sensor buffer)
latest_data = {
    "cars": 0,
    "bikes": 0,
    "buses": 0,
    "trucks": 0,
    "day": 0,
    "hour": 12
}

@app.route("/iot-data", methods=["POST"])
def receive_iot_data():
    global latest_data
    latest_data = request.json
    return jsonify({"status": "Data received successfully"})

@app.route("/get-latest", methods=["GET"])
def get_latest_data():
    return jsonify(latest_data)

if __name__ == "__main__":
    app.run(port=5000)
