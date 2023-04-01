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


def display_debug_window(image):
    cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
    hsv_feed = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    min_road = (0, 0, 85)
    max_road = (255, 3, 210)
    light_gray_mask = cv2.inRange(hsv_feed, min_road, max_road)
    ERODE_KERNEL_SIZE = 10
    BLUR_KERNEL_SIZE = 40
    LICENSE_AREA_THRESHOLD = 10e3
    eroded = cv2.erode(light_gray_mask, np.ones((ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE), np.uint8), iterations=1)
    blurred = cv2.blur(eroded, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE))
    eroded_again = cv2.erode(blurred, np.ones((ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE), np.uint8), iterations=1)
    result = eroded_again
    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        filled_areas = []
        aspect_ratios = []
        # Filter out contours that are too wide
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = rect[1][0]
            height = rect[1][1]
            if width == 0 or height == 0:
                aspect_ratio = 99
            elif height > width:
                aspect_ratio = height / width
            else:
                aspect_ratio = width / height
            aspect_ratios.append(aspect_ratio)
        contours = [contour for contour, aspect_ratio in zip(contours, aspect_ratios) if aspect_ratio < 2]
        # get largest square-like contour
        if len(contours) > 0:
            for contour in contours:
                current_contour_mask = np.zeros_like(light_gray_mask)
                cv2.drawContours(current_contour_mask, [contour], 0, 255, cv2.FILLED)
                filled_areas.append(cv2.countNonZero(current_contour_mask))
            maxFilledContour = contours[np.argmax(filled_areas)]
            rect = cv2.minAreaRect(maxFilledContour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            maxArea = cv2.contourArea(box)
            if maxArea > LICENSE_AREA_THRESHOLD:
                cv2.drawContours(hsv_feed,[box],0,(255,255,255), 10)
                cv2.putText(result, 'max area: ' + str(maxArea // 1000) + 'k', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    # display
    cv2.imshow('mask', cv2.resize(result, (hsv_feed.shape[1]//2, hsv_feed.shape[0]//2)))
    cv2.imshow('debug', cv2.resize(hsv_feed, (hsv_feed.shape[1]//2, hsv_feed.shape[0]//2)))
    cv2.moveWindow('debug', 1250, 10)
    cv2.waitKey(1)




# show debug window
def display_truck_mask(image):
    cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")

    hsv_feed = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    min_road = (0, 0, 85)
    max_road = (255, 10, 110)
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
        if totalArea > 70000:
            cv2.putText(mask, 'STOP', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20, cv2.LINE_AA)

    cv2.imshow('debug', cv2.resize(mask, (hsv_feed.shape[1]//2, hsv_feed.shape[0]//2)))
    cv2.moveWindow('debug', 1250, 10)
    cv2.waitKey(1)
    

# set up subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=display_debug_window)

# forever
rospy.spin()