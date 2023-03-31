import controller
import rospy
from enum import Enum
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from config import ASSETS_FOLDER
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Constants
# relative paths (inside ASSETS FOLDER)


# init ros node
rospy.init_node('Debugger', anonymous=True)

# show debug window
def display_debug_window(image):
    cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")

    hsv_feed = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    min_road = (0, 0, 70)
    max_road = (10, 10, 90)

    mask = cv2.inRange(hsv_feed, min_road, max_road)

    # show image
    cv2.imshow('debug', cv2.resize(mask, (hsv_feed.shape[1]//2, hsv_feed.shape[0]//2)))
    cv2.moveWindow('debug', 1250, 10)
    cv2.waitKey(1)
    

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=display_debug_window)

# forever
rospy.spin()