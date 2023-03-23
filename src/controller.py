from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

import rospy
import cv2
import numpy as np
import datetime
from enum import Enum
import tensorflow as tf
import os
import time


# Driving Mode



class Operating_Mode(Enum):
    MANUAL = 1,
    MODEL = 2
    TAKE_PICTURES = 3

class ControllerState(Enum):
    INIT = 4,
    DRIVE_OUTER_LOOP = 3,
    WAIT_FOR_PEDESTRIAN = 2,
    MANUAL_DRIVE = 1

# type of input image to model
class Image_Type(Enum):
    BGR = 1,
    GRAY = 2





class Controller:

    # ========= Initialization =========

    def __init__(self, operating_mode, image_save_location, image_type, start_snapshots, snapshot_freq, 
                 image_resize_factor, publisher, drive_diagonal, driving_model_path, linear_speed, angular_speed, color_converter):
        self.bridge = CvBridge()
        self.iters = 0
        self.vels = Twist()
        self.camera_feed = None
        self.image_save_location=image_save_location
        self.image_type = image_type
        self.start_snapshots = start_snapshots
        self.snapshot_freq = snapshot_freq
        self.image_resize_factor = image_resize_factor
        self.publisher=publisher
        self.drive_diagonal = drive_diagonal
        self.driving_model_path = driving_model_path
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.operating_mode = operating_mode
        self.state = ControllerState.INIT
        self.color_converter = color_converter
        self.driving_model=None
        
    def load_model(self):
        self.driving_model = tf.keras.models.load_model(self.driving_model_path)


    def step(self, data):
        self.iters+=1
        self.camera_feed = self.convert_image_topic_to_cv_image(data)
        # show video output
        # self.show_camera_feed(self.camera_feed)


        print('=== state: ', self.state)
        # display 
        cv2.imshow('video feed', self.downsample_image(self.camera_feed, 2))
        cv2.waitKey(1)
        
        # Jump to state
        if self.state == ControllerState.INIT:
            self.RunInitState()
        elif self.state == ControllerState.DRIVE_OUTER_LOOP:
            self.RunDriveOuterLoopState()
        elif self.state == ControllerState.MANUAL_DRIVE:
            self.RunManualDriveState()
        elif self.state == ControllerState.WAIT_FOR_PEDESTRIAN:
            self.RunWaitForPedestrianState()

    # ========= States ===============

    # Switch to drive state
    def RunInitState(self):
        # start outer loop if in Model mode
        if self.operating_mode == Operating_Mode.MODEL:
            self.state = ControllerState.DRIVE_OUTER_LOOP
        # start manual driving if in manual mode
        elif self.operating_mode == Operating_Mode.MANUAL or self.operating_mode == Operating_Mode.TAKE_PICTURES:
            self.state = ControllerState.MANUAL_DRIVE


    def RunDriveOuterLoopState(self):
        predicted_action = np.argmax(self.call_driving_model(self.camera_feed)) 
        if self.check_if_at_crosswalk():
            move = Twist() # don't move
            self.state = ControllerState.WAIT_FOR_PEDESTRIAN # state transition ? 
            print(f"state changed to {self.state}")
        else:
            move = self.convert_action_to_cmd_vel(predicted_action, self.drive_diagonal)
        self.publisher.publish(move)
        return


    def RunManualDriveState(self):
        # take pictures if neededs
        if self.operating_mode == Operating_Mode.TAKE_PICTURES and self.iters > self.start_snapshots and self.iters % self.snapshot_freq == 0:
            self.save_image(self.downsample_image(self.camera_feed, self.image_resize_factor, self.color_converter), str([self.vels.linear.x, self.vels.angular.z]) + str(datetime.datetime.now()))
            if self.iters % 100 == 0:
                print('image folder has ', len(os.listdir(self.image_save_location)), 'images')
        return


    
    def RunWaitForPedestrianState(self):
        # wait for pedestrian
        min_skin =(5,60,60)
        max_skin = (12,90,90)

        wait_flag = True
        hsv_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv_feed, min_skin, max_skin)

        skin_contours, skin_hierarchy = cv2.findContours(image=skin_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        skin_cntrs = []

        for c in skin_contours:
            if cv2.contourArea(c) > 10:
                skin_cntrs.append(c)

        if len(skin_cntrs) > 0:
            # define bounding rectangle
            x,y,w,h = cv2.boundingRect(skin_cntrs[0])
            cv2.rectangle(self.camera_feed,(x,y),(x+w,y+h),(0,255,0),2) 
            # check if at center:
            TOL_SKIN = 30
            for c in skin_cntrs:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
            if abs(cX- skin_mask.shape[1]//2) < TOL_SKIN:
                self.state = ControllerState.DRIVE_OUTER_LOOP
                print(f'state changed to {self.state}')

        cv2.imshow('mask', self.downsample_image(skin_mask, 2))
        cv2.waitKey(1)

        # return to drive outer loop




    # =========== Utilities =============

    def check_if_at_crosswalk(self):
        Y_exp_of_first_cntr = 600
        Y_exp_of_second_cntr = 410
        radius_of_tolerance = 50

        at_crosswalk_flag = False
        min_red = (0, 200, 170)   # lower end of blue
        max_red = (255, 255, 255)   # upper end of blue

        # Binary mask for blue pixels
        hsv_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_feed, min_red, max_red)

        # get contours
        red_contours, red_hierarchy = cv2.findContours(image=red_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        red_cntrs = []

        # only take contours with area > 5
        for c in red_contours:
            if cv2.contourArea(c) > 5:
                red_cntrs.append(c)

        # sort red counters based on area
        red_c = sorted(red_cntrs, key = lambda x : cv2.contourArea(x)) # sort by x value

        # draw bounding rectangles
        for i in range(len(red_cntrs)):
            # define bounding rectangle
            x,y,w,h = cv2.boundingRect(red_cntrs[i])
            cv2.rectangle(self.camera_feed,(x,y),(x+w,y+h),(0,255,255),2) 

        red_centroids = []
        if len(red_cntrs) == 2:
            for c in red_cntrs:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                red_centroids.append((cX, cY))

            # check if STOP
            if abs(red_centroids[0][1] - 650) < radius_of_tolerance and abs(red_centroids[1][1] - 410) < radius_of_tolerance:
                at_crosswalk_flag = True
                print("Reached crosswalk.")

        cv2.imshow('mask', self.downsample_image(red_mask, 2))
        cv2.waitKey(1)

        return at_crosswalk_flag


    def call_driving_model(self, camera_feed):
        # downsample
        camera_feed_gray =  self.downsample_image(camera_feed, self.image_resize_factor, color_converter=cv2.COLOR_BGR2GRAY)
        camera_feed_gray = tf.expand_dims(camera_feed_gray, 0) # expand dim 0
        softmaxes = tf.squeeze(self.driving_model(camera_feed_gray),0) 
        return softmaxes

    
    def convert_image_topic_to_cv_image(self, camera_topic):
        return self.bridge.imgmsg_to_cv2(camera_topic, 'bgr8')

    def convert_action_to_cmd_vel(self, action, drive_diagonal):
        out = Twist()
        if not drive_diagonal: # drive straight mode
            if action == 0:
                out.linear.x = 0.5
            elif action == 1:
                out.angular.z = 1.0 
            else:
                out.angular.z = -1.0
        else:
            out.linear.x = self.linear_speed
            if action == 1:
                out.angular.z = self.angular_speed
            elif action == 2:
                out.angular.z = -1 * self.angular_speed
        return out

    def read_camera_message(self, msg):
        return self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def store_velocities(self, vels_topic):
        self.vels = vels_topic

    def save_image(self, image, filename):
        print(' saving image', self.image_save_location + '/' + filename + '.jpg')
        cv2.imwrite(self.image_save_location + '/' + filename + '.png', image)

    def downsample_image(self, img, factor, color_converter=None):
        shape = img.shape
        resized_shape = (shape[1] // factor, shape[0] // factor)
        out =  cv2.resize(img, resized_shape, interpolation=cv2.INTER_AREA)
        if color_converter is not None:
            out = cv2.cvtColor(out, color_converter)
        return out
    
    def show_camera_feed(self, raw_camera_feed):
        raw_camera_feed = self.downsample_image(raw_camera_feed, 2) # downsample by 2
        labelled_video_feed = self.annotate_image(raw_camera_feed)
        cv2.imshow("Camera feed", labelled_video_feed)
        cv2.waitKey(1)

    def annotate_image(self, input_image):
        out = input_image.copy()
        velocity_array = [self.vels.linear.x, self.vels.linear.y, self.vels.linear.z, self.vels.angular.z]
        velocity_attributes_and_values = [s + ': ' + str(m) for s, m in zip(['x', 'y', 'z', 'angular'], velocity_array)]
        for i in range(len(velocity_attributes_and_values)):
            cv2.putText(img=out, text=velocity_attributes_and_values[i], org=(20,20+40*i), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(img=out, text="iters: " + str(self.iters), org=(150, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        return out




