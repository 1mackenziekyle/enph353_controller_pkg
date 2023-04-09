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
import time

# Constants
# relative paths (inside ASSETS FOLDER)


# init ros node
rospy.init_node('Mask Debugger', anonymous=True)


def display_debug_window(img):
    img = CvBridge().imgmsg_to_cv2(img, "bgr8")
    img = cv2.medianBlur(img,5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    uh = 5
    us = 5
    uv = 72
    lh = 0
    ls = 0
    lv = 48 
    lower_hsv = np.array([lh,ls,lv])
    upper_hsv = np.array([uh,us,uv])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    window_name = "HSV Calibrator"
    cv2.namedWindow(window_name)
    def nothing(x):
        print("Trackbar value: " + str(x))
        pass
    cv2.createTrackbar('UpperH',window_name,0,255,nothing)
    cv2.setTrackbarPos('UpperH',window_name, uh)
    cv2.createTrackbar('UpperS',window_name,0,255,nothing)
    cv2.setTrackbarPos('UpperS',window_name, us)
    cv2.createTrackbar('UpperV',window_name,0,255,nothing)
    cv2.setTrackbarPos('UpperV',window_name, uv)
    cv2.createTrackbar('LowerH',window_name,0,255,nothing)
    cv2.setTrackbarPos('LowerH',window_name, lh)
    cv2.createTrackbar('LowerS',window_name,0,255,nothing)
    cv2.setTrackbarPos('LowerS',window_name, ls)
    cv2.createTrackbar('LowerV',window_name,0,255,nothing)
    cv2.setTrackbarPos('LowerV',window_name, lv)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("Loaded images")
    while(1):
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        cv2.putText(mask,'Lower HSV: [' + str(lh) +',' + str(ls) + ',' + str(lv) + ']', (10,30), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
        cv2.putText(mask,'Upper HSV: [' + str(uh) +',' + str(us) + ',' + str(uv) + ']', (10,60), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
        cv2.imshow(window_name,mask)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        uh = cv2.getTrackbarPos('UpperH',window_name)
        us = cv2.getTrackbarPos('UpperS',window_name)
        uv = cv2.getTrackbarPos('UpperV',window_name)
        upper_blue = np.array([uh,us,uv])
        lh = cv2.getTrackbarPos('LowerH',window_name)
        ls = cv2.getTrackbarPos('LowerS',window_name)
        lv = cv2.getTrackbarPos('LowerV',window_name)
        upper_hsv = np.array([uh,us,uv])
        lower_hsv = np.array([lh,ls,lv])
        time.sleep(0.1)

    
# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=display_debug_window)

# forever
rospy.spin()