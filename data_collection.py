import requests
import time
from datetime import datetime

NODEMCU_DEVICES = [
    {"id": 1, "hostname": "nodemcu-sensor1.local"},
    {"id": 2, "hostname": "nodemcu-sensor2.local"},
    {"id": 3, "hostname": "nodemcu-sensor3.local"},
    {"id": 4, "hostname": "nodemcu-sensor4.local"},
    {"id": 5, "hostname": "nodemcu-sensor5.local"},
    {"id": 6, "hostname": "nodemcu-sensor6.local"},
    {"id": 7, "hostname": "nodemcu-sensor7.local"},
    {"id": 8, "hostname": "nodemcu-sensor8.local"},
    {"id": 9, "hostname": "nodemcu-sensor9.local"},
    {"id": 10, "hostname": "nodemcu-sensor10.local"},
    {"id": 11, "hostname": "nodemcu-sensor11.local"},
]

PORT = 80

def send_request(base_url, endpoint, method="GET", data=None):
    url = f"{base_url}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, data=data)
        else:
            print("Unsupported HTTP method")
            return None

        print(f"HTTP Response code: {response.status_code}")
        print(f"Response content: {response.text}")
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def sync_time(device):
    current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    print(f"Syncing time for NodeMCU {device['id']} ({device['hostname']}): {current_time}")
    base_url = f"http://{device['hostname']}:{PORT}"
    send_request(base_url, "/sync", method="POST", data={"time": current_time})

def start_data_collection(device):
    print(f"Starting data collection for NodeMCU {device['id']} ({device['hostname']})...")
    base_url = f"http://{device['hostname']}:{PORT}"
    send_request(base_url, "/con")

def stop_data_collection(device):
    print(f"Stopping data collection for NodeMCU {device['id']} ({device['hostname']})...")
    base_url = f"http://{device['hostname']}:{PORT}"
    send_request(base_url, "/coff")

if __name__ == '__main__':
    try:
        for device in NODEMCU_DEVICES:
            sync_time(device)
            start_data_collection(device)

        print("Data collection is running indefinitely. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping data collection...")

    finally:
        for device in NODEMCU_DEVICES:
            stop_data_collection(device)

        print("Data collection stopped for all devices.")
