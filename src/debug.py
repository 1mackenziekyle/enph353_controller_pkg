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
import numpy as np

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
    h = mask.shape[0]
    mask[:6*h//10, :] = 0 # zero out top 3/4 
    mask[7*h//10:, :] = 0 # zero out bottom 1/4
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.stack([mask] * 3, axis=-1)
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)
    if len(contours) > 0:
        biggestContour = max(contours, key=cv2.contourArea)
        totalArea = sum([cv2.contourArea(contour) for contour in contours])
        cv2.putText(mask, 'max area: ' + str(totalArea // 1000) + 'k', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if totalArea > 80000:
            cv2.putText(mask, 'STOP', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20, cv2.LINE_AA)

    cv2.imshow('debug', cv2.resize(hsv_feed, (hsv_feed.shape[1]//2, hsv_feed.shape[0]//2)))
    cv2.moveWindow('debug', 1250, 10)
    cv2.waitKey(1)
    

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=display_debug_window)

# forever
rospy.spin()