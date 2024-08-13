import requests
import time
from datetime import datetime


NODEMCU_HOSTNAME = "nodemcu-sensor.local"  
PORT = 80

BASE_URL = f"http://{NODEMCU_HOSTNAME}:{PORT}"

START_ENDPOINT = "/con"
STOP_ENDPOINT = "/coff"
SYNC_ENDPOINT = "/sync"

def send_request(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
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

def sync_time():
    current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    print(f"Syncing time: {current_time}")
    send_request(SYNC_ENDPOINT, method="POST", data={"time": current_time})


def start_data_collection():
    print("Starting data collection...")
    send_request(START_ENDPOINT)

def stop_data_collection():
    print("Stopping data collection...")
    send_request(STOP_ENDPOINT)

if __name__ == '__main__':
    sync_time()
    start_data_collection()
    time.sleep(10)
    stop_data_collection()
    print("Data collection stopped.")
