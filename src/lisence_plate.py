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
def draw_rect(x,y,w,h, image):
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),6)
def show(image, label='img'):
    cv2.imshow(label, image)
    cv2.waitKey(1)
def get_aspect_ratio(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return h/w
def slice(contour, first): 
    x,y,w,h = cv2.boundingRect(contour)
    w1 = w//2
    if(first):
        return x,y,w1,h
    else:
        return x+w1,y,w1,h


def label_license_plate(data):
    image = bridge.imgmsg_to_cv2(data, 'bgr8')
    hsv_feed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Find bounding box for license plate and parking number
    min_grey = (0, 0, 97)
    max_grey = (0,0, 210)
    min_plate_grey = (100, 0, 85)
    max_plate_grey = (130, 70, 180)
    min_blue = (110, 135, 90)
    max_blue = (125, 210, 200)
    min_blue_letters = (110, 100, 85)
    max_blue_letters = (125, 245, 197)
    max_blue_letters_model = (125,245,200)
    min_blue_letters_model = (110,100,90)
    # mask on each color
    blue_mask = cv2.inRange(hsv_feed, min_blue, max_blue)
    outer_mask = cv2.inRange(hsv_feed, min_grey, max_grey) 
    plate_mask = cv2.inRange(hsv_feed, min_plate_grey, max_plate_grey)
    # combine masks
    lisence = cv2.bitwise_or(outer_mask, plate_mask)
    lisence = cv2.bitwise_or(lisence, blue_mask)
    # blur and threshold to get rid of lines
    lisence_mask = cv2.GaussianBlur(lisence, (31, 31), cv2.BORDER_DEFAULT)
    _, lisence_mask = cv2.threshold(lisence_mask, 190, 255, cv2.THRESH_BINARY)
    # lisence_mask_stacked = np.stack([lisence_mask, lisence_mask, lisence_mask], axis=-1)
    # find biggest c    ontour
    contours, hierarchy = cv2.findContours(image=lisence_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    if (len(contours) > 0):
        contours = sorted(contours, key = lambda x : cv2.contourArea(x)) # sort by area
        biggest_area = cv2.contourArea(contours[-1])
        if 100000 > biggest_area > 15000:
            #print(biggest_area)
            x,y,w,h = cv2.boundingRect(contours[-1])
            if x > 0 and x + w < lisence_mask.shape[1] : 
                # read license plate
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),6) 
                lisence_plate_img = image[y:y+h, x:x+w]
                lisence_plate_img_hsv = cv2.cvtColor(lisence_plate_img, cv2.COLOR_BGR2HSV)
                print(lisence_plate_img_hsv.shape)
                lisence_plate_chars = cv2.inRange(lisence_plate_img_hsv, min_blue_letters, max_blue_letters)
                lisence_chars_stacked = np.stack([lisence_plate_chars, lisence_plate_chars, lisence_plate_chars], axis=-1)
                char_contours, hierarchy = cv2.findContours(image=lisence_plate_chars, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                char_contours = [c for c in char_contours if cv2.contourArea(c) > 5]
                char_contours = [c for c in char_contours if get_aspect_ratio(c) < 4]
                char_contours = sorted(char_contours, key = lambda x : cv2.contourArea(x)) # sort by area
                print(len(char_contours))
                num_contours = 0;
                for contour in char_contours:
                    print(get_aspect_ratio(contour))
                    if (get_aspect_ratio(contour) < .4 and cv2.contourArea(contour) > 300):
                        x1,y1,w1,h1 = slice(contour, first=True)
                        x2,y2,w2,h2 = slice(contour, first=False)
                        draw_rect(x1, y1, w1, h1, lisence_chars_stacked)
                        draw_rect(x2, y2, w2, h2, lisence_chars_stacked)
                        char = lisence_plate_img_hsv[y1:y1 + h1, x1: x1 + w1]
                        char_s = char[:, :, 1]
                        char_s = cv2.resize(char_s, (20,20))
                        show(char_s, label='char' + str(num_contours))
                        num_contours += 1
                        char2 = lisence_plate_img_hsv[y2:y2 + h2, x2: x2 + w2]
                        char_s2 = char2[:, :, 1]
                        char_s2 = cv2.resize(char_s2, (20,20))
                        show(char_s2, label='char' + str(num_contours))
                        num_contours += 1
                    else:
                        x1,y1,w1,h1 = cv2.boundingRect(contour)
                        draw_rect(x1, y1, w1, h1, lisence_chars_stacked)
                        char = lisence_plate_img_hsv[y1:y1 + h1, x1: x1 + w1]
                        char_s = char[:, :, 1]
                        char_s = cv2.resize(char_s, (20,20))
                        print(char_s.shape)
                        #char = cv2.inRange(char, min_blue_letters_model, max_blue_letters_model)
                        show(char_s, label='char' + str(num_contours))
                        num_contours += 1
                show(lisence_chars_stacked, label='lisence')
    
#initialize rospy subscriber
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback= label_license_plate)

#forever
rospy.spin()
