import controller
import string
import rospy
import time
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
CHARACTER_RECOGNITION_MODEL_PATH = ASSETS_FOLDER + 'models/char_recog_resize_2'





# relative paths (inside ASSETS FOLDER)
class LicensePlateDetection:
    def __init__(self, model_path, cooldown=1.0):
        self.bridge = CvBridge()
        self.model = tf.keras.models.load_model(model_path)
        self.cooldown = cooldown
        self.chars = list(string.ascii_uppercase + string.digits)
        self.ctoi = {c:i for i,c in enumerate(self.chars)}
        self.itoc = {i:c for i,c in enumerate(self.chars)}
        self.guesses = []
        self.license_plate_read_timestamp = 10000
        self.current_parking_spot = 1
        self.license_plate_publisher = rospy.Publisher('/license_plate', String, queue_size=1)



    def character_forwards_pass(self, img):
        preds = []
        for index in range(0,4):
            input = tf.expand_dims(tf.expand_dims(img[index], -1), 0)
            probs = self.model(input)
            probs = tf.squeeze(probs, axis = 0)
            if index < 2:
                pred = str(self.itoc[np.argmax(probs[:26])])
            else:
                pred = str(self.itoc[26 + np.argmax(probs[26:])])
            preds.append(pred)          
        return preds


    def draw_rect(self, x,y,w,h, image):
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),6)


    def show(self, image, label='img'):
        cv2.imshow(label, image)
        cv2.waitKey(1)

    

    def get_min_aspect_ratio(self, contour):
        rect = cv2.minAreaRect(contour)
        (x,y), (w, h), angle = rect
        if (not w == 0 and not h == 0):
            return min(w,h)/max(w,h)
        else:
            return 0 


    def get_area_ar_ratio(self, contour):
        return self.get_min_aspect_ratio(contour)/cv2.contourArea(contour)


    def get_aspect_ratio(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        return h/w


    def slice(self, contour, first): 
        x,y,w,h = cv2.boundingRect(contour)
        w1 = w//2
        if(first):
            return x,y,w1,h
        else:
            return x+w1,y,w1,h


    def find_x_value(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        return x

    
    def downsample(self, image, factor):
        resized_shape = (int(image.shape[1] / factor), int(image.shape[0] / factor))
        return cv2.resize(image, resized_shape, interpolation=cv2.INTER_AREA)
    

    def most_frequent(self, List):
        dict = {}
        count, itm = 0, ''
        for item in reversed(List):
            dict[item] = dict.get(item, 0) + 1
            if dict[item] >= count :
                count, itm = dict[item], item
        return(itm)

    def most_common(self, list):
        most_common_item = max(set(list), key=list.count)
        if list.count(most_common_item) == 1:
            return list[-1]
        else:
            return most_common_item


    def char_array_to_string(self, s):
        # initialization of string to ""
        new = ""
        # traverse in the string
        for x in s:
            new += x
        # return string
        return new


    def label_license_plate(self, data):
        # update timestamp
        if len(self.guesses) > 0 and time.time() - self.license_plate_read_timestamp > self.cooldown:
            best_guess = self.most_common(self.guesses)
            print(f'Best guess out of {len(self.guesses)}: ', best_guess)
            # publish license plate
            self.license_plate_publisher.publish(str(f'Team8,multi21,{self.current_parking_spot},{best_guess}'))
            self.guesses = []
            self.current_parking_spot += 1
        # mask out everything but the license plate box
        camera_feed = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        hsv_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2HSV)
        # Find bounding box for license plate and parking number
        min_grey = (0, 0, 97)
        max_grey = (0,0, 210)
        min_plate_grey = (100, 0, 85)
        max_plate_grey = (130, 70, 180)
        min_blue = (110, 135, 90)
        max_blue = (125, 210, 200)
        min_blue_letters = (110, 100, 85)
        max_blue_letters = (125, 245, 197)
        blue_mask = cv2.inRange(hsv_feed, min_blue, max_blue)
        outer_mask = cv2.inRange(hsv_feed, min_grey, max_grey) 
        plate_mask = cv2.inRange(hsv_feed, min_plate_grey, max_plate_grey)
        # combine masks
        lisence = cv2.bitwise_or(outer_mask, plate_mask)
        lisence = cv2.bitwise_or(lisence, blue_mask)
        # blur and threshold to get rid of lines
        lisence_mask = cv2.GaussianBlur(lisence, (31, 31), cv2.BORDER_DEFAULT)
        _, lisence_mask = cv2.threshold(lisence_mask, 190, 255, cv2.THRESH_BINARY)
        # find biggest contour
        contours, hierarchy = cv2.findContours(image=lisence_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        # only take contours based on area and aspect ratio
        # calculate aspect ratio: 
        contours = [c for c in contours if 2.0 > cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3] > 0.4 
                                        and cv2.contourArea(c) > 10000 # TODO: MAX AREA
                                        and cv2.contourArea(c) / (cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3]) > 0.7]
        # ========= DEBUGGING LISENCE PLATE CONDITIONS =========
        if len(contours) > 0:
            maxContour = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(maxContour)
            if x > 0 and x + w  < camera_feed.shape[1]-1: # ensure not on edge of image
                stacked = np.stack((lisence_mask,)*3, axis=-1)
                cv2.rectangle(stacked,(x-1,y-1),(x+w+1,y+h+1),(0, 255, 0),2)
                cv2.putText(stacked, 'A: ' + str(round(cv2.contourArea(maxContour) / 1000, 2)) + 'k', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                aspect_ratio = w/h
                cv2.putText(stacked, 'w/h: ' + str(round(aspect_ratio, 3)), (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.putText(stacked, '% filled: ' + str(round(cv2.contourArea(maxContour) / (w*h), 2 )), (x,y-70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.putText(stacked, 'x: ' + str((cv2.boundingRect(maxContour)[0] - stacked.shape[1]//2)/stacked.shape[1]), (x,y-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.imshow('mask', self.downsample(stacked, 1.5))
                # crop license plate
                license_plate_image = camera_feed[y:y+h, x:x+w]
                license_plate_image_hsv = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2HSV)
                license_plate_blue_mask = cv2.inRange(license_plate_image_hsv, min_blue_letters, max_blue_letters)
                letter_contours, letter_hierarchy = cv2.findContours(image=license_plate_blue_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                letter_contours = [c for c in letter_contours if cv2.contourArea(c) > 2]
                
                letter_contours = sorted(letter_contours, key=lambda c: abs(cv2.boundingRect(c)[0] - license_plate_blue_mask.shape[1]//2))[:4] # take 4 central contours
                
                letter_contours = sorted(letter_contours, key=lambda c: cv2.boundingRect(c)[0]) # sort by x position
                if len(letter_contours) == 0:
                    return
                # letter images
                letter_boxes = []
                for c in letter_contours:
                    x,y,w,h = cv2.boundingRect(c)
                    letter_boxes.append((x,y,w,h)) # x,y,w,h
                # Case: 2 big contours
                if len(letter_contours) == 2:
                    # split contours into 2 
                    x1,y1,w1,h1 = cv2.boundingRect(letter_contours[0]) # letters
                    x2,y2,w2,h2 = cv2.boundingRect(letter_contours[1]) # numbers
                    letter_boxes.insert(2, (x2+w2//2,y2,w2//2,h2)) # second number
                    letter_boxes.insert(1, (x1+w1//2,y1,w1//2,h1)) # second letter
                    letter_boxes[0] = (x1,y1,w1//2,h1) # first letter
                    letter_boxes[2] = (x2,y2,w2//2,h2) # first number
                if len(letter_contours) == 3:
                    indexLargest = np.argmax([cv2.contourArea(c) for c in letter_contours])
                    x,y,w,h = cv2.boundingRect(letter_contours[indexLargest])
                    letter_boxes.insert(indexLargest+1, (x+w//2,y,w//2,h)) # right half
                    letter_boxes[indexLargest]= (x,y,w//2,h) # left half
                    # TODO: If model isnt good at letters with blackspace around it, resize image to fit contour once split into 2 to get rid of black space
                guess = ""
                for i, (x,y,w,h) in enumerate(letter_boxes):
                    model_input = tf.expand_dims(tf.expand_dims(cv2.resize(license_plate_image_hsv[y:y+h,x:x+w,1], (20,20)),0),-1)
                    if i < 2:
                        guess += self.itoc[np.argmax(tf.squeeze(self.model(model_input),0)[:26])]
                    else:
                        guess += self.itoc[26 + np.argmax(tf.squeeze(self.model(model_input),0)[26:])]
                print(f'P{self.current_parking_spot} Guess: ', guess)
                self.guesses.append(guess)
                self.license_plate_read_timestamp = time.time()
                cv2.imshow('Letters!', np.concatenate([cv2.resize(license_plate_image_hsv[y:y+h,x:x+w,1], (20,20)) for x,y,w,h in letter_boxes], axis=1))
                cv2.imshow('License plate', self.downsample(license_plate_blue_mask, 0.5))
                cv2.waitKey(1)
                # print(f'License plate found, {len(letter_contours)} contours.')
        else: 
            cv2.imshow('mask', self.downsample(lisence_mask, 1.5))
        cv2.waitKey(1)
        # ========= DEBUGGING LISENCE PLATE CONDITIONS =========




# init ros node
rospy.init_node('Liscence_plate_detection', anonymous=True)
licensePlateReader = LicensePlateDetection(model_path=CHARACTER_RECOGNITION_MODEL_PATH)
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=licensePlateReader.label_license_plate)
rospy.spin()
