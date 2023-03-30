from sensor_msgs.msg import Image
from std_msgs.msg import String
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
import matplotlib.pyplot as plt

class Operating_Mode(Enum):
    MANUAL = 1,
    MODEL = 2,
    TAKE_PICTURES = 3,
    SADDLE = 4,
    TAKE_PICTURES_LISENCE = 5

class ControllerState(Enum):
    INIT = 4,
    DRIVE_OUTER_LOOP = 3,
    WAIT_FOR_PEDESTRIAN = 2,
    MANUAL_DRIVE = 1,




"""
Notes for use: 

The Robot class works by attaching the 'step' function to the /camera_raw message in ROS,
so that the controller takes one 'step' or action after each camera image is processed.

After each /camera_raw message is published, the controller.step function is called.
Then, the controller jumps into RunCurrentState function, which calls
a Run{X}State function and the robot executes whatever is in that Run{X}State function.

Where {X} is the current state.
"""




# ======================== Configuration
# ========== Debugging
DEBUG_RED_MASK = False
DEBUG_SKIN_MASK = False
SHOW_MODEL_OUTPUTS = False
DEBUG_HSV_OUTPUT = True
DEBUG_LISENCE_MASK = True 
# ========== Saving images
IMAGE_SAVE_FOLDER = 'images/outer_lap/Liscence_plate'
SNAPSHOT_FREQUENCY = 2
COLOR_CONVERTER = None
RESIZE_FACTOR = 1

# ========== Loading Model
DRIVING_MODEL_LOAD_FOLDER = 'models/outer_lap/5convlayers/saddle/saddle0'

# ========== Operating
OPERATING_MODE = Operating_Mode.MANUAL
LINEAR_SPEED = 0.3645
ANGULAR_SPEED = 1.21





class Controller:
    def __init__(self, operating_mode, image_save_location, start_snapshots, snapshot_freq, 
                 image_resize_factor, cmd_vel_publisher, license_plate_publisher, driving_model_path, linear_speed, angular_speed, color_converter):
        """
        Initialize variables and load model for Controller object.
        """
        self.bridge = CvBridge()
        self.iters = 0
        self.vels = Twist()
        self.camera_feed = None
        self.image_save_location=image_save_location
        self.start_snapshots = start_snapshots
        self.snapshot_freq = snapshot_freq
        self.image_resize_factor = image_resize_factor
        self.cmd_vel_publisher=cmd_vel_publisher
        self.license_plate_publisher=license_plate_publisher
        self.driving_model_path = driving_model_path
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.operating_mode = operating_mode
        self.state = ControllerState.INIT
        self.color_converter = color_converter
        if self.operating_mode is not Operating_Mode.TAKE_PICTURES:
            self.driving_model = tf.keras.models.load_model(self.driving_model_path)
        self.take_pictures = False



    def step(self, data):
        """
        Enter RunState() for current state

        Update and display camera feed

        (Temporary) Send a message of value -1 to '/license_plate' topic after 100 iterations
        for time trials
        """
        self.iters+=1
        self.camera_feed = self.convert_image_topic_to_cv_image(data)
        self.show_camera_feed(self.camera_feed)
        if self.iters == 100: # TODO: REMOVE AFTER TIME TRIALS
            self.license_plate_publisher.publish(str('Team8,multi21,-1,XR58'))
        print('=== state: ', self.state)
        # Jump to state
        self.RunCurrentState()



    def RunCurrentState(self):
        """
        Jump into state according to self.state variable
        """
        if self.state == ControllerState.INIT:
            self.RunInitState()
        elif self.state == ControllerState.DRIVE_OUTER_LOOP:
            self.RunDriveOuterLoopState()
        elif self.state == ControllerState.MANUAL_DRIVE:
            self.RunManualDriveState()
        elif self.state == ControllerState.WAIT_FOR_PEDESTRIAN:
            self.RunWaitForPedestrianState()



    def RunInitState(self):
        """
        Start the competition timer and enter either 
        the model-driving state or self-driving state 
        """
        time.sleep(1) # wait 1 second for ros to initialize completely
        self.license_plate_publisher.publish(str('Team8,multi21,0,XR58'))
        if self.operating_mode == Operating_Mode.MODEL:
            self.state = ControllerState.DRIVE_OUTER_LOOP
        # start manual driving if in manual mode
        elif self.operating_mode == Operating_Mode.MANUAL or self.operating_mode == Operating_Mode.TAKE_PICTURES or self.operating_mode == Operating_Mode.SADDLE:
            self.state = ControllerState.MANUAL_DRIVE



    def RunDriveOuterLoopState(self):
        if self.check_if_at_crosswalk():
            move = Twist() # don't move
            self.state = ControllerState.WAIT_FOR_PEDESTRIAN # state transition ? 
            print(f"state changed to {self.state}")
            time.sleep(0.1) # TODO: Better way to do this ?
        else:
            softmaxes = self.call_driving_model(self.camera_feed)
            predicted_action = np.argmax(softmaxes) 
            move = self.convert_action_to_cmd_vel(predicted_action)
        self.cmd_vel_publisher.publish(move)
        return



    def RunManualDriveState(self):
        print(self.vels)
        print('Take Pictures: ', self.take_pictures)
        hsv_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2HSV)
        if self.operating_mode == Operating_Mode.TAKE_PICTURES: 
            self.save_labelled_image()
        elif self.operating_mode == Operating_Mode.TAKE_PICTURES_LISENCE:
            self.save_labelled_image_lisence()
        elif self.operating_mode == Operating_Mode.SADDLE:
            human_action = self.convert_cmd_vels_to_action(self.vels.linear.x, self.vels.angular.z)
            model_action = np.argmax(self.call_driving_model(self.camera_feed))
            if human_action != model_action:
                self.save_labelled_image()
        if DEBUG_HSV_OUTPUT:
            cv2.imshow('HSV Image', self.downsample_image(hsv_feed, 2))
            cv2.moveWindow('HSV Image', 10, 500)
            cv2.waitKey(1)
        if DEBUG_LISENCE_MASK:
            min_grey = (0, 0, 97)
            max_grey = (0,0, 200)
            lisence_mask = cv2.inRange(hsv_feed, min_grey, max_grey)
            lisence_contours, lisence_heierarchy = cv2.findContours(image=lisence_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            lisence_contours_sorted = sorted(lisence_contours, key = lambda x : cv2.contourArea(x)) # sort by area
            x,y,w,h = cv2.boundingRect(lisence_contours_sorted[-1])
            lisence_mask_stacked = np.stack([lisence_mask, lisence_mask, lisence_mask], axis=-1)
            cv2.rectangle(lisence_mask_stacked,(x,y),(x+w,y+h),(0,0,255),2) 
            cv2.imshow('Lisence Mask', self.downsample_image(lisence_mask_stacked,2))
            #cv2.moveWindow('Lisence Mask', 10, 500)
            cv2.waitKey(1)

             
        return


    
    def RunWaitForPedestrianState(self):
        TOL_SKIN = 30
        min_skin =(5,60,60)
        max_skin = (12,90,90)
        hsv_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv_feed, min_skin, max_skin)
        skin_contours, skin_hierarchy = cv2.findContours(image=skin_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        skin_cntrs = []
        for c in skin_contours:
            if cv2.contourArea(c) > 10:
                skin_cntrs.append(c)
        if len(skin_cntrs) > 0:
            x,y,w,h = cv2.boundingRect(skin_cntrs[0])
            cv2.rectangle(self.camera_feed,(x,y),(x+w,y+h),(0,255,0),2) 
            for c in skin_cntrs:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
            if abs(cX- skin_mask.shape[1]//2) < TOL_SKIN:
                self.state = ControllerState.DRIVE_OUTER_LOOP
        if DEBUG_SKIN_MASK:
            cv2.imshow('Pedestrian Mask', self.downsample_image(skin_mask, 2))
            cv2.waitKey(1)



# ==================== Utility Functions =======================
    def self_labelled_image_lisence(self):
        if self.iters > self.start_snapshots:
            if self.iters % self.snapshot_freq == 0:
                if self.take_pictures:
                    self.save_image(self.downsample_image(self.camera_feed, self.image_resize_factor, self.color_converter), str([self.vels.linear.x, self.vels.angular.z]) + str(datetime.datetime.now()))

    def save_labelled_image(self):
        if self.iters > self.start_snapshots:
            if self.iters % self.snapshot_freq == 0:
                if abs(self.vels.linear.x + self.vels.angular.z) > 0 and self.vels.linear.x >= 0 and self.take_pictures:
                    self.save_image(self.downsample_image(self.camera_feed, self.image_resize_factor, self.color_converter), str([self.vels.linear.x, self.vels.angular.z]) + str(datetime.datetime.now()))

    def check_if_at_crosswalk(self):
        """
        Returns True/False of whether robot thinks it is at the crosswalk and should stop.
        """
        Y_exp_0 = 650
        Y_exp_1 = 410
        radius_of_tolerance = 40
        min_red = (0, 200, 170)   # lower end of blue
        max_red = (255, 255, 255)   # upper end of blue
        hsv_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_feed, min_red, max_red)
        red_contours, red_hierarchy = cv2.findContours(image=red_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        red_cntrs = []
        for c in red_contours:
            if cv2.contourArea(c) > 5:
                red_cntrs.append(c)
        red_c = sorted(red_cntrs, key = lambda x : cv2.contourArea(x)) # sort by x value
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
            if abs(red_centroids[0][1] - Y_exp_0) < radius_of_tolerance and abs(red_centroids[1][1] - Y_exp_1) < radius_of_tolerance:
                return True # exit WaitForPedestrian and resume driving
        if DEBUG_RED_MASK:
            red_mask = np.stack([np.zeros_like(red_mask), np.zeros_like(red_mask), red_mask], axis=-1)
            cv2.imshow('Pedestrian Mask', self.downsample_image(red_mask, 2))
            cv2.moveWindow('Pedestrian Mask', 10, 500)
            cv2.waitKey(1)
        return False



    def call_driving_model(self, camera_feed):
        """
        Forward passes a cv2 camera image through the driving network and returns
        softmax probabalistic outputs.
        Ex: (Image of left turning road) ---> [0.2, 0.7, 0.1]
                                                F    L    R 
        """
        camera_feed_gray =  self.downsample_image(camera_feed, self.image_resize_factor, color_converter=cv2.COLOR_BGR2GRAY) # downsample
        camera_feed_gray = tf.expand_dims(camera_feed_gray, 0) # expand batch dim = 1
        softmaxes = tf.squeeze(self.driving_model(camera_feed_gray),0) # Squeeze output shape: (1, N) --> (N)
        return softmaxes
 

    
    def convert_image_topic_to_cv_image(self, camera_topic):
        return self.bridge.imgmsg_to_cv2(camera_topic, 'bgr8')



    def convert_action_to_cmd_vel(self, action):
        """
        Maps an action (0, 1, or 2) to a Twist() object to publish to ROS network
        """
        out = Twist()
        out.linear.x = self.linear_speed
        if action == 1:
            out.angular.z = self.angular_speed
        elif action == 2:
            out.angular.z = -1 * self.angular_speed
        return out



    def convert_cmd_vels_to_action(self, linear_x, angular_z):
        """
        Requires that input is not (0,0) (no action)
        """
        if angular_z == 0:
            return 0
        elif angular_z > 0:
            return 1
        else: 
            return 2



    def read_camera_message(self, msg):
        return self.bridge.imgmsg_to_cv2(msg, 'bgr8')



    def store_velocities(self, vels_topic):
        self.vels = vels_topic
        if vels_topic.linear.z > 0.0:
            self.take_pictures = not self.take_pictures
            print('toggled picture taking.')



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
        cv2.moveWindow("Camera feed", 10, 100)
        cv2.waitKey(1)



    def annotate_image(self, input_image):
        out = input_image.copy()
        velocity_array = [self.vels.linear.x, self.vels.linear.y, self.vels.linear.z, self.vels.angular.z]
        velocity_attributes_and_values = [s + ': ' + str(m) for s, m in zip(['x', 'y', 'z', 'angular'], velocity_array)]
        for i in range(len(velocity_attributes_and_values)):
            cv2.putText(img=out, text=velocity_attributes_and_values[i], org=(20,20+40*i), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(img=out, text="iters: " + str(self.iters), org=(150, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(img=out, text="Mode: " + str(self.operating_mode.name), org=(450, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        return out