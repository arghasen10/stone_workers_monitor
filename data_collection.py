import requests
import time

# Replace with your NodeMCU IP address
NODEMCU_IP = "10.5.20.136"  # Update this with the correct IP if needed
PORT = 80

# Base URL for HTTP requests
BASE_URL = f"http://{NODEMCU_IP}:{PORT}"

# Define endpoints for enabling and disabling data collection
START_ENDPOINT = "/con"
STOP_ENDPOINT = "/coff"

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

def start_data_collection():
    print("Starting data collection...")
    send_request(START_ENDPOINT)

def stop_data_collection():
    print("Stopping data collection...")
    send_request(STOP_ENDPOINT)

if __name__ == '__main__':
    start_data_collection()
    time.sleep(10)
    stop_data_collection()
    print("Data collection stopped.")
