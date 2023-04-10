import controller
import string
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
CHARACTER_RECOGNITION_MODEL_PATH = ASSETS_FOLDER + 'models/char_recog_resize'
# relative paths (inside ASSETS FOLDER)
bridge = CvBridge()
model = tf.keras.models.load_model(CHARACTER_RECOGNITION_MODEL_PATH)

chars = list(string.ascii_uppercase + string.digits)
ctoi = {c:i for i,c in enumerate(chars)}
itoc = {i:c for i,c in enumerate(chars)}

# init ros node
rospy.init_node('Liscence_plate_detection', anonymous=True)
def character_forwards_pass(img):
    preds = []
    for index in range(0,4):
        input = tf.expand_dims(tf.expand_dims(img[index], -1), 0)
        probs = model(input)
        probs = tf.squeeze(probs, axis = 0)
        if index < 2:
            pred = str(itoc[np.argmax(probs[:26])])
        else:
            pred = str(itoc[26 + np.argmax(probs[26:])])
        preds.append(pred)          
    return preds
def draw_rect(x,y,w,h, image):
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),6)
def show(image, label='img'):
    cv2.imshow(label, image)
    cv2.waitKey(1)
def get_min_aspect_ratio(contour):
    rect = cv2.minAreaRect(contour)
    (x,y), (w, h), angle = rect
    if (not w == 0 and not h == 0):
        return min(w,h)/max(w,h)
    else:
        return 0 
def get_area_ar_ratio(contour):
    return get_min_aspect_ratio(contour)/cv2.contourArea(contour)
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

def find_x_value(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return x

def label_license_plate(data):
    image = bridge.imgmsg_to_cv2(data, 'bgr8')
    hsv_feed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    min_grey = (0, 0, 97)
    max_grey = (0,0, 210)
    min_plate_grey = (100, 0, 85)
    max_plate_grey = (130, 70, 180)
    min_blue = (110, 135, 90)
    max_blue = (125, 210, 200)
    min_blue_letters = (110, 100, 85)
    max_blue_letters = (125, 245, 197)
    #max_blue_letters_model = (125,245,200)
    #min_blue_letters_model = (110,100,90)
    blue_mask = cv2.inRange(hsv_feed, min_blue, max_blue)
    outer_mask = cv2.inRange(hsv_feed, min_grey, max_grey) 
    plate_mask = cv2.inRange(hsv_feed, min_plate_grey, max_plate_grey)
    lisence = cv2.bitwise_or(outer_mask, plate_mask)
    lisence = cv2.bitwise_or(lisence, blue_mask)
    lisence_mask = cv2.GaussianBlur(lisence, (31, 31), cv2.BORDER_DEFAULT)
    _, lisence_mask = cv2.threshold(lisence_mask, 190, 255, cv2.THRESH_BINARY)
    lisence_mask_stacked = np.stack([lisence_mask, lisence_mask, lisence_mask], axis=-1)
    contours, hierarchy = cv2.findContours(image=lisence_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if get_min_aspect_ratio(c) > .5]
    contours = [c for c in contours if cv2.contourArea(c) > 4000]
    contours = [c for c in contours if get_area_ar_ratio(c) < .0001]
    if (len(contours) > 0):
        contours = sorted(contours, key = lambda x : cv2.contourArea(x)) # sort by area
        biggest_area = cv2.contourArea(contours[-1])
        if (biggest_area < 15000):
           contours = [c for c in contours if get_min_aspect_ratio(c) > .75]
        #print(biggest_area)
        if (len(contours)> 0):
            x,y,w,h = cv2.boundingRect(contours[-1])
            draw_rect(x,y,w,h, lisence_mask_stacked)
            show(lisence_mask_stacked, 'lisence mask')
            if 100000 > biggest_area:
                #print(biggest_area)
                x,y,w,h = cv2.boundingRect(contours[-1])
                if x > 0 and x + w < lisence_mask.shape[1] : 
                    # read license plate
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),6) 
                    lisence_plate_img = image[y:y+h, x:x+w]
                    lisence_plate_img_hsv = cv2.cvtColor(lisence_plate_img, cv2.COLOR_BGR2HSV)
                    #print(lisence_plate_img_hsv.shape)
                    lisence_plate_chars = cv2.inRange(lisence_plate_img_hsv, min_blue_letters, max_blue_letters)
                    lisence_chars_stacked = np.stack([lisence_plate_chars, lisence_plate_chars, lisence_plate_chars], axis=-1)
                    char_contours, hierarchy = cv2.findContours(image=lisence_plate_chars, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                    char_contours = [c for c in char_contours if cv2.contourArea(c) > 5]
                    char_contours = [c for c in char_contours if get_aspect_ratio(c) < 4]
                    char_contours = sorted(char_contours, key = lambda x : cv2.contourArea(x)) # sort by area
                    #print(len(char_contours))
                    num_contours = 0;
                    char_contours.reverse() 
                    char_img = []
                    char_img_x_val = []
                    if (len(char_contours) >= 4):
                        slice_bool = False
                        num_slices = 0
                        del char_contours[4:]
                    elif(len(char_contours) <= 1):
                        slice_bool = False
                        num_slices = 0
                    else:
                        slice_bool = True
                        num_slices = 4 - len(char_contours)
                    for contour in char_contours:
                        #print(get_aspect_ratio(contour))
                        if (num_slices > 0 and slice_bool):
                            x1,y1,w1,h1 = slice(contour, True)
                            x2,y2,w2,h2 = slice(contour, False)
                            draw_rect(x1, y1, w1, h1, lisence_chars_stacked)
                            draw_rect(x2, y2, w2, h2, lisence_chars_stacked)
                            char = lisence_plate_img_hsv[y1:y1 + h1, x1: x1 + w1]
                            char_s = char[:, :, 1]
                            char_s = cv2.resize(char_s, (20,20))
                            char_img.append(char_s)
                            char_img_x_val.append(x1)
                            #show(char_s, label='char' + str(num_contours))
                            num_contours += 1
                            char2 = lisence_plate_img_hsv[y2:y2 + h2, x2: x2 + w2]
                            char_s2 = char2[:, :, 1]
                            char_s2 = cv2.resize(char_s2, (20,20))
                            char_img.append(char_s2)
                            char_img_x_val.append(x2)
                           # show(char_s2, label='char' + str(num_contours))
                            num_contours += 1
                            num_slices = num_slices - 1
                        else:
                            x1,y1,w1,h1 = cv2.boundingRect(contour)
                            draw_rect(x1, y1, w1, h1, lisence_chars_stacked)
                            char = lisence_plate_img_hsv[y1:y1 + h1, x1: x1 + w1]
                            char_s = char[:, :, 1]
                            char_s = cv2.resize(char_s, (20,20))
                            char_img.append(char_s)
                            char_img_x_val.append(x1)
                            #show(char_s, label='char' + str(num_contours))
                            num_contours += 1
                    imgs, x_vals = zip(*sorted(zip(char_img, char_img_x_val), key= lambda b:b[1]))
                    imgs = list(imgs)
                    out = np.concatenate(imgs, axis=1)
                    show(out, 'full plate')

                    pred = character_forwards_pass(imgs)
                    print(pred)      
                    #cv2.moveWindow("char0", 650, 50)
                    #cv2.moveWindow("char1", 950, 50)
                    #cv2.moveWindow("char2", 950, 200)
                    #cv2.moveWindow("char3", 650, 200)
                    show(lisence_chars_stacked, label='lisence')


    
#initialize rospy subscriber
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback= label_license_plate)

#forever
rospy.spin()
