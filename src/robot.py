#! /usr/bin/env python3

import controller
import rospy
from enum import Enum
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from config import ASSETS_FOLDER
import cv2

# Constants
# relative paths (inside ASSETS FOLDER)


# init ros node
rospy.init_node('robot', anonymous=True)

# initialize controller object
robot = controller.Controller(
    operating_mode=controller.OPERATING_MODE,
    image_save_location=ASSETS_FOLDER + controller.IMAGE_SAVE_FOLDER,
    start_snapshots=20,
    snapshot_freq=controller.SNAPSHOT_FREQUENCY,
    image_resize_factor=controller.RESIZE_FACTOR,
    cmd_vel_publisher=rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1),
    license_plate_publisher=rospy.Publisher('/license_plate', String, queue_size=1),
    outer_loop_driving_model_path=ASSETS_FOLDER + controller.OUTER_LOOP_DRIVING_MODEL_PATH,
    inner_loop_driving_model_path=ASSETS_FOLDER + controller.INNER_LOOP_DRIVING_MODEL_PATH,
    outer_loop_linear_speed=controller.OUTER_LOOP_LINEAR_SPEED,
    outer_loop_angular_speed=controller.OUTER_LOOP_ANGULAR_SPEED,
    inner_loop_linear_speed=controller.INNER_LOOP_LINEAR_SPEED,
    inner_loop_angular_speed=controller.INNER_LOOP_ANGULAR_SPEED,
    color_converter=controller.COLOR_CONVERTER
    )

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=robot.step)
rospy.Subscriber('R1/cmd_vel', Twist, callback=robot.store_velocities)

# forever
rospy.spin()
