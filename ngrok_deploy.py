import os
import firebase_admin
from firebase_admin import credentials, db
import subprocess
import time
import requests
import signal

from datetime import datetime as dt

# Start your Node.js app
python_app_directory = 'C:\\Users\\neurog\\Desktop\\AutogenRag'
python_app = subprocess.Popen(['python', 'ollama_chatbot.py'], cwd=python_app_directory)

# Give some time for your Node.js app to start
time.sleep(10)
# Firebase configuration
firebase_config = {
    "apiKey": "AIzaSyDwI2UUaxnfXBGbYXc7gQhJ8iEWRc93_5c",
    "authDomain": "tweets-zero-theorem.firebaseapp.com",
    "databaseURL": "https://tweets-zero-theorem-default-rtdb.firebaseio.com",
    "projectId": "tweets-zero-theorem",
    "storageBucket": "tweets-zero-theorem.appspot.com",
    "messagingSenderId": "676722815795",
    "appId": "1:676722815795:web:e86d97cb7fe84e7e968446",
    "measurementId": "G-6HBMBVTB61"
}

current_dir = os.getcwd()
# Firebase credentials
json_path = os.path.join(current_dir, "key.json")
cred = credentials.Certificate(json_path)

# Initialize Firebase app
firebase_admin.initialize_app(cred, firebase_config)

# Get a reference to the Firebase Realtime Database
ref = db.reference('backend-api-link-chatbot')

# Start Ngrok to expose the Node.js app to the public
ngrok_executable_path = 'ngrok'  # Replace with the actual path to ngrok.exe
ngrok_process = subprocess.Popen([ngrok_executable_path, 'http',  str(5001)])

ngrok_url = None

try:
    # Keep checking for Ngrok tunnel until available
    while not ngrok_url:
        try:
            ngrok_info = requests.get('http://localhost:4040/api/tunnels').json()
            tunnels = ngrok_info['tunnels']

            if tunnels:
                ngrok_url = tunnels[0]['public_url']
                ref.set({"link": ngrok_url})
                print(f'Ngrok URL: {ngrok_url}')
            else:
                time.sleep(1)
        except requests.exceptions.ConnectionError as e:
            print(f'Error: {e}')
            # Handle the ConnectionError, e.g., wait and retry
            time.sleep(1)


    # # Keep the script running
    # while True:
    #     if dt.utcnow().minute == 57:
    #         print("System is on exist mode to restart properly")
    #         exit(0)
    #     time.sleep(1)

except KeyboardInterrupt:
    # If the user interrupts the script (e.g., by pressing Ctrl+C), terminate Ngrok, the Node.js app, and MySQL server
    ngrok_process.terminate()
    node_app_process.terminate()
    mysql_server_process.terminate()
    os.killpg(os.getpgid(ngrok_process.pid), signal.SIGTERM)  # Terminate the Ngrok process group
    print("Ngrok, Node.js app, and MySQL server terminated.")
