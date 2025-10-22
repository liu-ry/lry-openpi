## Folding T-shirt Application

This application is designed for model inference in the t-shirt folding scenario. 

- The model is trained based on the OpenPi0 model (refer to: https://github.com/Physical-Intelligence/openpi)

- The robot that we are using is YAM i2rt (refer to: https://github.com/i2rt-robotics/i2rt)

### Prerequisite
Create a conda envrionment, install i2rt and realsense driver.
1. conda create -n robot python=3.10
2. conda activate robot
3. cd i2rt_manipulation 
4. pip install -e .
5. cd ../openpi-client
6. pip install -e .
7. pip install pyrealsense2



### Steps to run

1. IP Address and Port Configuration:

You need to modify the IP address and port number to match the machine on which your model is running:
```
host: str = "192.168.3.5"  # Replace with your model-running machine's IP
port: int = 8040           # Replace with the corresponding port number
```

2. Offline Performance Validation: 

Before deploying the model on the actual dual-arm robot, it is recommended to verify the model's performance using offline data (located in the example_data directory). If the Ground Truth (GT) aligns well with the predicted actions, the offline validation is considered passed.
```
python server_client.py
```

3. Deployment on Real Robot: 

To deploy the model and run the task on the physical dual-arm robot, follow these steps:
```
# Navigate to the script directory
cd i2rt_manipulation/scripts

# Reset all CAN bus connections
sh reset_all_can.sh

# Start the robot action execution program
python robot_action_execution.py
```

4. Task Termination: 

If you need to stop the ongoing task, first click on the displayed image (to ensure the window is in focus), then press the `q` key on your keyboard.


