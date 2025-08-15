# Drone-VLM-Detection

This project combines a **DJI Tello drone** with a **SmolVLM vision-language model** to automatically scan environments and **detect physical security risks**, such as exposed credentials, unattended devices, or suspicious objects. The drone autonomously traverses a grid, capturing images and analyzing them in real time for potential threats.

# Writeup
Drone-VLM-Detection is an autonomous security audit system that pairs a DJI Tello drone with a lightweight SmolVLM vision-language model to detect real-world security risks. The drone systematically scans an environment in a grid or spiral pattern, capturing images and analyzing them locally in real time for indicators such as exposed credentials, unlocked devices, unattended USB drives, visible keycards, and other potential vulnerabilities.

By automating physical threat detection, the system reduces the time, labor, and expertise required for on-site security sweeps. It operates offline, protecting sensitive environments from cloud exposure, and delivers instant, location-tagged findings for faster incident response. This approach enables organizations to continuously monitor high-risk areas, improve audit coverage, and address vulnerabilities before they can be exploited, bridging the gap between physical security vulnerabilities and real-world implementation.

# Drone Setup

Clone this repo. 

Create python env: ``` python -m venv drone-vlm ```

Activate python env: ``` source drone-vlm/bin/activate ``` 

Install the requirements.txt: ``` pip install -r requirements.txt ``` 

The drone should broadcast its own wifi network named Tello-[ID Something]

Connect to the wifi


## Download the VLM 
``` python download_model.py ```


## Run the VLM Script 
``` python drone_combined_path.py ``` 



> **Example threats detected:**  
> - Unlocked computers  
> - USB drives  
> - Laptops  
> - Keys  
> - Name tags  
> - Codes or passwords on whiteboards


##  How It Works

- **1. Autonomous Flight:**  
  The Tello drone traverses a room in a spiral/grid pattern, streaming video back to your computer.

- **2. Onboard Threat Detection:**  
  Every five frames a local instance of [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) is used to caption and extract objects of interest.

- **3. Real-Time Reporting:**  
  Detected threats are printed in the console, and logged by location.

---

##  Tools & Technologies

- **DJI Tello** (`djitellopy`): Lightweight programmable drone with live video streaming.
- **SmolVLM**: Lightweight vision-language model for local, offline inference.
- **PyTorch**: Model execution backend.
- **Transformers**: Model loading and processing.
- **OpenCV**: Video frame manipulation and UI display.
- **Pillow**: Image conversions for model compatibility.

## Demo:

[Video](https://youtu.be/syawe8rNSzQ)

## For Debugging:


## Drone light info:

- yellow charging 
- green connected 
- red low battery, issue with connection

## Default Drone IP/port

192.168.10.1. Port: '8889'.


## Debugging if port is in use 


Check the port process: ``` lsof -i :8889 ```
Kill that process: ``` kill -9 [id] ```

## Contributing
We welcome contributions to this project! To contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push your branch to your fork.
4. Submit a pull request with a detailed description of your changes.


## License
This project is licensed under the MIT License.

## Authors and Acknowledgments
This project was developed by the intern team at Everwatch Corporation:

- Myra Cropper
- Jonathan Mei
- Sachin Ashok
- Kyle Simon
- Matthew Lessler
- Izaiah Davis
- Quinn Dunnigan
- Julia Calvert
- Julie Ochs
- Alyssa Miller
- Bentley Cech

Mentored by David Culver.


## Contact Information
For questions or issues, please open an issue in the repository or contact the authors.

