import controller
import rospy
from enum import Enum
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from config import ASSETS_FOLDER
from cv_bridge import CvBridge
import cv2
import tensorflow as tf
# Constants
CHARACTER_RECOGNITION_MODEL_PATH = ASSETS_FOLDER + 'models/character_recognition/char_recog_cheaper'
# relative paths (inside ASSETS FOLDER)
bridge = CvBridge()
model = tf.keras.models.load_model(CHARACTER_RECOGNITION_MODEL_PATH)

# init ros node
rospy.init_node('Liscence_plate_detection', anonymous=True)

def label_license_plate(data):
    image = bridge.imgmsg_to_cv2(data, 'bgr8')
    hsv_feed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    min_grey = (0, 0, 97)
    max_grey = (0,0, 210)
    min_plate_grey = (100, 0, 85)
    max_plate_grey = (130, 70, 180)
    min_blue = (110, 135, 90)
    max_blue = (125, 210, 200)
    blue_mask = cv2.inRange(hsv_feed, min_blue, max_blue)
    outer_mask = cv2.inRange(hsv_feed, min_grey, max_grey) 
    plate_mask = cv2.inRange(hsv_feed, min_plate_grey, max_plate_grey)
    lisence = cv2.bitwise_or(outer_mask, plate_mask)
    lisence = cv2.bitwise_or(lisence, blue_mask)
    lisence_mask = cv2.GaussianBlur(lisence, (31, 31), cv2.BORDER_DEFAULT)
    _, lisence_mask = cv2.threshold(lisence_mask, 190, 255, cv2.THRESH_BINARY)
    lisence_mask_stacked = np.stack([lisence_mask, lisence_mask, lisence_mask], axis=-1)
    contours, hierarchy = cv2.findContours(image=lisence_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    if (len(contours) > 0):
        contours = sorted(contours, key = lambda x : cv2.contourArea(x)) # sort by area
        biggest_area = cv2.contourArea(contours[-1])
        if 100000 > biggest_area > 15000:
            x,y,w,h = cv2.boundingRect(contours[-1])
            if x > 0 and x + w < lisence_mask.shape[1] : 
                # read license plate
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),6) 
                lisence_plate_img = image[y:y+h, x:x+w]
                lisence_plate_img_hsv = cv2.cvtColor(lisence_plate_img, cv2.COLOR_BGR2HSV)
                lisence_plate_chars = cv2.inRange(lisence_plate_img_hsv, min_blue, max_blue)
                cv2.putText(image, 'License Plate', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'License Plate detected', (20, image.shape[0]-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 4)
                cv2.imshow('License Plate', image[y:y+h, x:x+w])
                cv2.waitKey(1)
                cv2.moveWindow('License Plate', 650, 100)
                for i in range(6):
                    print(np.argmax(model(np.expand_dims(np.expand_dims(lisence_plate_chars[:20, :15], 0), axis=-1))), " ", end="")
                print()

#initialize rospy subscriber
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback= label_license_plate)

#forever
rospy.spin()
