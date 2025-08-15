# Drone-VLM-Detection

yay drone setup

Create python env: ``` python -m venv drone-vlm ```

Activate python env: ``` source drone-vlm/bin/activate ``` 

Install the requirements.txt: ``` pip install -r requirements.txt ``` 

The drone should broadcast its own wifi network named Tello-[ID Something]

Connect to the wifi

You can run the test script to make sure the drone is connected: ``` python drone_test.py ``` 

## Download the VLM 
``` python download_model.py ```


## Run the VLM Script 
- May need to change the between using either cuda, mps, or cpu 
``` python drone_vlm.py ``` 


## Drone light info:

- yellow charging 
- green connected 
- red low battery, issue with connection

## Default Drone IP/port

192.168.10.1. Port: '8889'.


## Debugging if port is in use 


Check the port process: ``` lsof -i :8889 ```
Kill that process: ``` kill -9 1234 ```


