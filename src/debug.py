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
def display_truck_mask(image):
    cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
    hsv_feed = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    min_truck = (0, 0, 48)
    max_truck = (5, 5, 62)
    mask = cv2.inRange(hsv_feed, min_truck, max_truck)
    mask = cv2.erode(mask, np.ones((2,2), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        maxContour = max(contours, key=cv2.contourArea)
        truckSize = cv2.contourArea(maxContour)
        cv2.drawContours(mask, [maxContour], -1, (0,255,0), 3)
    else:
        truckSize = 0
    mask = np.stack((mask,)*3, axis=-1)
    print('Truck Size: ' + str(truckSize))
    cv2.imshow('Truck Mask', cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2)))
    cv2.waitKey(1)
    
def display_red_mask(img):
    img = CvBridge().imgmsg_to_cv2(img, "bgr8")
    min_red = (0, 200, 170)   # lower end of blue
    max_red = (255, 255, 255)   # upper end of blue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_red, max_red)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if len(contours) > 0:
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out,(x,y),(x+w,y+h),255,6)
            cv2.putText(out, 'A: ' + str(cv2.contourArea(c)), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
            cv2.putText(out, 'w: ' + str(w), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
    cv2.imshow('Red Mask', cv2.resize(out, (out.shape[1]//2, out.shape[0]//2)))
    cv2.waitKey(1)


def waste_time(image):
    """
    Takes approx 50ms to run
    """
    for i in range(100):
        for j in range(200):
            print(i**2 / (j+1)**0.5)

    
# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=display_red_mask)

# forever
rospy.spin()