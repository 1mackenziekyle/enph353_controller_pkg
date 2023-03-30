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

    gray_min = (0, 0, 90)
    gray_max = (255, 10, 130)

    gray_mask = cv2.inRange(hsv_feed, gray_min, gray_max)

    contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find max contour
    if len(contours) != 0:
        # draw in blue the contours that were founded
        
        cv2.drawContours(hsv_feed, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(hsv_feed,(x,y),(x+w,y+h),(0,0,255),2)
    # show image
    cv2.imshow('debug', cv2.resize(hsv_feed, (hsv_feed.shape[1]//2, hsv_feed.shape[0]//2)))
    cv2.waitKey(1)
    

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=display_debug_window)

# forever
rospy.spin()