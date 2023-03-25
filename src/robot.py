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
IMAGE_SAVE_FOLDER = 'images/outer_lap/dagger1'
DRIVING_MODEL_LOAD_FOLDER = 'models/outer_lap/5convlayers/10000images_and_1000recentering'
OPERATING_MODE = controller.Operating_Mode.TAKE_PICTURES

# image saving
COLOR_CONVERTER = cv2.COLOR_BGR2GRAY
RESIZE_FACTOR = 20

LINEAR_SPEED = 0.3645
ANGULAR_SPEED = 1.21

SNAPSHOT_FREQUENCY = 1 # TODO: CHANGE BACK TO 1


# init ros node
rospy.init_node('robot', anonymous=True)

# initialize controller object
robot = controller.Controller(
    operating_mode=OPERATING_MODE,
    image_save_location=ASSETS_FOLDER + IMAGE_SAVE_FOLDER,
    image_type=controller.Image_Type.GRAY,
    start_snapshots=100,
    snapshot_freq=SNAPSHOT_FREQUENCY,
    image_resize_factor=RESIZE_FACTOR,
    cmd_vel_publisher=rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1),
    license_plate_publisher=rospy.Publisher('/license_plate', String, queue_size=1),
    drive_diagonal=True,
    driving_model_path=ASSETS_FOLDER + DRIVING_MODEL_LOAD_FOLDER,
    linear_speed=LINEAR_SPEED,
    angular_speed=ANGULAR_SPEED,
    color_converter=COLOR_CONVERTER
    )

# load model if in model mode

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=robot.step)
rospy.Subscriber('R1/cmd_vel', Twist, callback=robot.store_velocities)

# forever
rospy.spin()