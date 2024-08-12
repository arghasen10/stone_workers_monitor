from datetime import datetime
from flask import Flask, request

app = Flask(__name__)

fp = open("sensor_data.txt", "a+")

@app.route('/sensor', methods=['POST'])
def sensor():
    # Get form data
    sensor_data = request.form.to_dict()
    print(sensor_data)
    sensor_data['time'] = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    fp.write(str(sensor_data) + '\n')
    fp.flush()
    
    # You can process the sensor data here as needed
    return "Data received", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6001, debug=True)
