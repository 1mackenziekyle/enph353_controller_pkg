# Self-Driving Robot in ROS

This project implements a basic self-driving robot using the Robot Operating System (ROS). The robot autonomously navigates using a series of custom-trained imitation learning driving models (or manually), captures images of license plates, reads them using computer vision and neural nets, and logs relevant data. The main script initializes and runs the robot with specific configurations for movement, image processing, and data storage.

## Features
- Autonomously navigates using inner and outer loop driving models.
- If capturing data for training models, captures and saves images from a camera feed
- Detects license plates, reads them using a custom-trained character recognition model
- Publishes and subscribes to relevant ROS topics for controlling and monitoring.
- Logs and stores movement commands and detected license plates.

## Letter Recognition
- The jupyter notebook generates images of skewed, rotated, and blurred characters, and trains a simple MLP to infer the characters on a license plate.


![image](https://github.com/user-attachments/assets/e59e37cc-a12f-42e7-9982-a30ef38f2ddc)
