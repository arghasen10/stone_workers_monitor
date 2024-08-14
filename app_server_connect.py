import subprocess
from flask import Flask, request , jsonify
from datetime import datetime
import os
import signal
from data_collection import sync_time, start_data_collection, stop_data_collection, NODEMCU_DEVICES

app = Flask(__name__)
fp = open("sensor_data.txt", "a+")

data_collection_process = None


@app.route('/ControllerOn', methods=['POST'])
def handle_ControllerOn():
    data = request.get_json()
    print(f"Received message: {data['message']}")
    return jsonify({ "message": "Controller is on"}),200
    
    

@app.route('/SYNCESP', methods=['POST'])
def handle_SYNCESP():
    data = request.get_json()
    print(f"Received message: {data['message']}")
    for device in NODEMCU_DEVICES:
        sync_time(device)
    return jsonify({"status": "success", "message": "Sync ESP is done"}),200
    


@app.route('/StartmmwaveDataCollection', methods=['POST'])
def handle_StartmmwaveDataCollection():
    data = request.get_json()
    
    global data_collection_process
    if data['message'] == 'Start mmWave DataCollection':
        try:
            data_collection_process = subprocess.Popen(['python', '/home/rajib/stone_workers_monitor/radar_data_read.py'])
            return jsonify({ "message": "mmWave data collection started.."}), 200
        
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"message":"data collection faliure"}),400    
        

    

@app.route('/StopmmwaveDataCollection', methods=['POST'])
def handle_StopmmwaveDataCollection():
    global data_collection_process
    data = request.get_json()

    if data['message'] == "stop mmwave datacollection":
        try:
            if data_collection_process:
                data_collection_process.terminate()
                data_collection_process.wait()

                # Reset the variable
                data_collection_process = None

                return jsonify({"status": "success", "message": "mmWave data collection stopped"}), 200
            else:
                return jsonify({"status": "failure", "message": "No data collection process is running."}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "failure", "message": "Invalid command."}), 400
    
   

@app.route('/StartEspDataCollection', methods=['POST'])
def handle_StartEspDataCollection():
    data = request.get_json()
    print(f"Received message: {data['message']}")
    for device in NODEMCU_DEVICES:
        start_data_collection(device)
    return jsonify({"status": "success", "message": "ESP data collection started.."}),200
    
    
    

@app.route('/StopEspDataCollection', methods=['POST'])
def handle_StopEspDataCollection():
    data = request.get_json()
    print(f"Received message: {data['message']}")
    for device in NODEMCU_DEVICES:
        stop_data_collection(device)
    return jsonify({"status": "success", "message": "ESP data collection stopped"}),200
    

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
    app.run(host='0.0.0.0', port=5001)
