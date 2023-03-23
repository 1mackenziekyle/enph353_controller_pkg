import controller
import rospy
from enum import Enum
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from config import ASSETS_FOLDER

# Constants
# relative paths (inside ASSETS FOLDER)
IMAGE_SAVE_FOLDER = 'images/outer_lap/gray_manual_recenter'
DRIVING_MODEL_LOAD_FOLDER = 'models/outer_lap/gray_diagonal_driving1'
OPERATING_MODE = controller.Operating_Mode.MODEL
COLOR_CONVERTER = None
LINEAR_SPEED = 0.3645
ANGULAR_SPEED = 1.21


# init ros node
rospy.init_node('robot', anonymous=True)

# initialize controller object
robot = controller.Controller(
    operating_mode=OPERATING_MODE,
    image_save_location=ASSETS_FOLDER + IMAGE_SAVE_FOLDER,
    image_type=controller.Image_Type.GRAY,
    start_snapshots=100,
    snapshot_freq=2,
    image_resize_factor=20,
    publisher=rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1),
    drive_diagonal=True,
    driving_model_path=ASSETS_FOLDER + DRIVING_MODEL_LOAD_FOLDER,
    linear_speed=LINEAR_SPEED,
    angular_speed=ANGULAR_SPEED,
    color_converter=COLOR_CONVERTER
    )

# load model if in model mode
if robot.operating_mode is controller.Operating_Mode.MODEL:
    robot.load_model()

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=robot.step)
rospy.Subscriber('R1/cmd_vel', Twist, callback=robot.store_velocities)

# forever
rospy.spin()
