import controller
import rospy
from enum import Enum
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from config import ASSETS_FOLDER

# Constants



# ============ Start Loop =============
rospy.init_node('robot', anonymous=True)

# initialize controller object
robot = controller.Controller(
    driving_mode=controller.Driving_Mode.TAKE_PICTURES,
    image_save_location=ASSETS_FOLDER + 'outer_lap_gray',
    start_snapshots=100,
    snapshot_freq=1,
    image_resize_factor=20,
    publisher=rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1),
    model_path='./assets/models/drive_outer_loop/',
    linear_speed=1.0,
    angular_speed=1.0
    )
if robot.driving_mode is controller.Driving_Mode.MODEL:
    robot.load_model()

# subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=robot.step)
rospy.Subscriber('R1/cmd_vel', Twist, callback=robot.read_velocities)
# forever
rospy.spin()