# Self-Driving Robot in ROS

This project implements a basic self-driving robot using the Robot Operating System (ROS). The robot autonomously navigates using a series of custom-trained imitation learning driving models (or manually), captures images of license plates, reads them using computer vision and neural nets, and logs relevant data. The main script initializes and runs the robot with specific configurations for movement, image processing, and data storage.

## Features
- Autonomously navigates using inner and outer loop driving models.
- If capturing data for training models, captures and saves images from a camera feed
- Detects license plates, reads them using a custom-trained character recognition model
- Publishes and subscribes to relevant ROS topics for controlling and monitoring.
- Logs and stores movement commands and detected license plates.
