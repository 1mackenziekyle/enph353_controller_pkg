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
    image_type=controller.Image_Type.GRAY,
    start_snapshots=20,
    snapshot_freq=controller.SNAPSHOT_FREQUENCY,
    image_resize_factor=controller.RESIZE_FACTOR,
    cmd_vel_publisher=rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1),
    license_plate_publisher=rospy.Publisher('/license_plate', String, queue_size=1),
    driving_model_path=ASSETS_FOLDER + controller.DRIVING_MODEL_LOAD_FOLDER,
    linear_speed=controller.LINEAR_SPEED,
    angular_speed=controller.ANGULAR_SPEED,
    color_converter=controller.COLOR_CONVERTER
    )

# load model if in model mode

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=robot.step)
rospy.Subscriber('R1/cmd_vel', Twist, callback=robot.store_velocities)

# forever
rospy.spin()