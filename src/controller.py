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
    INIT                    = 1,
    DRIVE_OUTER_LOOP        = 2,
    DRIVE_INNER_LOOP        = 3,
    WAIT_FOR_PEDESTRIAN     = 4,
    WAIT_FOR_TRUCK          = 5,
    END                     = 6,
    MANUAL_DRIVE            = 7,
class Image_Type(Enum):
    BGR = 1,
    GRAY = 2




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
DEBUG_HSV_OUTPUT = False
DEBUG_LISENCE_MASK = False
# ========== Saving images
IMAGE_SAVE_FOLDER = 'images/outer_lap/final/saddle7'
SNAPSHOT_FREQUENCY = 2
COLOR_CONVERTER = cv2.COLOR_BGR2GRAY
RESIZE_FACTOR = 20

# ========== Operating 
OPERATING_MODE = Operating_Mode.MODEL
TEST_INNER_LOOP = False

# ========== Model Settings
# OUTER_LOOP_LINEAR_SPEED = 0.3645
# OUTER_LOOP_ANGULAR_SPEED = 1.21
OUTER_LOOP_LINEAR_SPEED = 0.5
OUTER_LOOP_ANGULAR_SPEED = 2.14 
INNER_LOOP_LINEAR_SPEED = 0.266
INNER_LOOP_ANGULAR_SPEED = 1.0
OUTER_LOOP_DRIVING_MODEL_PATH = 'models/outer_lap/5convlayers/final/saddle7'
# OUTER_LOOP_DRIVING_MODEL_PATH = 'models/outer_lap/5convlayers/final/base10000'
INNER_LOOP_DRIVING_MODEL_PATH = 'models/inner_lap/first/base10000'



class Controller:
    def __init__(self, operating_mode, image_save_location, start_snapshots, snapshot_freq, 
                 image_resize_factor, cmd_vel_publisher, license_plate_publisher, outer_loop_driving_model_path, 
                 inner_loop_driving_model_path, outer_loop_linear_speed, outer_loop_angular_speed, inner_loop_linear_speed, 
                 inner_loop_angular_speed, color_converter):
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
        self.outer_loop_driving_model_path = outer_loop_driving_model_path
        self.inner_loop_driving_model_path = inner_loop_driving_model_path
        self.outer_loop_linear_speed = outer_loop_linear_speed
        self.outer_loop_angular_speed = outer_loop_angular_speed
        self.inner_loop_linear_speed = inner_loop_linear_speed
        self.inner_loop_angular_speed = inner_loop_angular_speed
        self.operating_mode = operating_mode
        self.state = ControllerState.INIT
        self.color_converter = color_converter
        if self.operating_mode is not Operating_Mode.TAKE_PICTURES and self.operating_mode is not Operating_Mode.MANUAL:
            self.outer_loop_driving_model = tf.keras.models.load_model(self.outer_loop_driving_model_path)
            self.inner_loop_driving_model = tf.keras.models.load_model(self.inner_loop_driving_model_path)
        self.take_pictures = False
        self.truck_passed = False
        self.prev_time_ms = time.time()
        self.done = False
        self.license_plates = {} # key: parking spot string, value: license plate string (e.g. 'P1': 'QX12')



    def step(self, data):
        """
        Enter RunState() for current state

        Update and display camera feed
        """
        start_time = time.time()
        self.iters+=1
        self.camera_feed = self.convert_image_topic_to_cv_image(data)
        if self.iters == 400 and self.operating_mode == Operating_Mode.MODEL:
            self. state = ControllerState.DRIVE_INNER_LOOP # TODO: REMOVE WHEN LICENSE PLATES DONE
        if self.iters == 850 and self.operating_mode == Operating_Mode.MODEL:
            self.state = ControllerState.END
        print('=== state: ', self.state, '=== Time between loop: ', int((time.time() - self.prev_time_ms) * 1000))
        self.prev_time_ms = time.time()
        # Jump to state
        self.RunCurrentState()
        # if self.operating_mode is Operating_Mode.MODEL: 
        #     self.label_license_plate(self.camera_feed)
        self.show_camera_feed(self.camera_feed)
        print(self.state, 'Loop time: ', int((time.time() - start_time) * 1000), 'time between loops: ', int((time.time() - self.prev_time_ms) * 1000))
        self.prev_time_ms = time.time()

        # TODO: REMOVE
        # time.sleep(0.04)
        # 40 ms seems to be the maximum delay between cmd_vel messages without causing the robot to leave track4



    def RunCurrentState(self):
        """
        Jump into state according to self.state variable
        """
        if self.state == ControllerState.INIT:
            self.RunInitState()
        elif self.state == ControllerState.DRIVE_OUTER_LOOP:
            self.RunDriveOuterLoopState()
        elif self.state == ControllerState.DRIVE_INNER_LOOP:
            self.RunDriveInnerLoopState()
        elif self.state == ControllerState.WAIT_FOR_PEDESTRIAN:
            self.RunWaitForPedestrianState()
        elif self.state == ControllerState.WAIT_FOR_TRUCK:
            self.RunWaitForTruckState()
        elif self.state == ControllerState.END:
            self.RunEndState()
        elif self.state == ControllerState.MANUAL_DRIVE:
            self.RunManualDriveState()



    def RunInitState(self):
        """
        Start the competition timer and enter either 
        the model-driving state or self-driving state 
        """
        self.license_plate_publisher.publish(str('Team8,multi21,0,XR58'))
        if self.operating_mode == Operating_Mode.MODEL:
            if TEST_INNER_LOOP:
                self.state = ControllerState.DRIVE_INNER_LOOP
            else:
                self.state = ControllerState.DRIVE_OUTER_LOOP
        # start manual driving if in manual mode
        elif self.operating_mode == Operating_Mode.MANUAL or self.operating_mode == Operating_Mode.TAKE_PICTURES or self.operating_mode == Operating_Mode.SADDLE:
            self.state = ControllerState.MANUAL_DRIVE
        time.sleep(1) # wait 1 second for ros to initialize completely
        return



    def RunDriveOuterLoopState(self):
        if self.check_if_at_crosswalk():
            move = Twist() # don't move
            self.state = ControllerState.WAIT_FOR_PEDESTRIAN # state transition ? 
            print(f"state changed to {self.state}")
            time.sleep(0.1) # TODO: Better way to do this ?
        else:
            softmaxes = self.call_outer_loop_driving_model(self.camera_feed)
            predicted_action = np.argmax(softmaxes) 
            move = self.convert_action_to_cmd_vel(predicted_action)
        self.cmd_vel_publisher.publish(move)
        cv2.waitKey(1)

        return

    

    def RunDriveInnerLoopState(self):
        TRUCK_STOP_CONTOUR_AREA = 70000
        # check for truck
        hsv_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2HSV)
        min_road = (0, 0, 70)
        max_road = (10, 10, 90)
        road_mask = cv2.inRange(hsv_feed, min_road, max_road)
        if not self.truck_passed:
            road_mask_cropped = road_mask.copy()
            h = road_mask.shape[0]
            road_mask_cropped[:6*h//10, :] = 0 # zero out top 3/4 
            road_mask_cropped[7*h//10:, :] = 0 # zero out bottom 1/4
            contours, hierarchy = cv2.findContours(road_mask_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                totalArea = sum([cv2.contourArea(contour) for contour in contours])
                # cv2.putText(mask, 'max area: ' + str(totalArea // 1000) + 'k', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if totalArea > TRUCK_STOP_CONTOUR_AREA:
                    # cv2.putText(mask, 'STOP', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20, cv2.LINE_AA)
                    self.state = ControllerState.WAIT_FOR_TRUCK
                    self.cmd_vel_publisher.publish(Twist()) # Stop moving
                    return
        softmaxes = self.call_inner_loop_driving_model(self.camera_feed)
        predicted_action = np.argmax(softmaxes)
        move = self.convert_action_to_cmd_vel(predicted_action)
        self.cmd_vel_publisher.publish(move)
        cv2.imshow('Hazard Detection', self.downsample_image(road_mask, 2))
        return



    def RunEndState(self):
        """
        End the competition timer and enter the end state
        """
        if not self.done:
            self.license_plate_publisher.publish(str('Team8,multi21,-1,XR58'))
            final_move = Twist()
            self.cmd_vel_publisher.publish(final_move)
            cv2.destroyWindow('Hazard Detection')
            self.done = True
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
            model_action = np.argmax(self.call_outer_loop_driving_model(self.camera_feed))
            if human_action != model_action:
                self.save_labelled_image()
        if DEBUG_HSV_OUTPUT:
            cv2.imshow('HSV Image', self.downsample_image(hsv_feed, 2))
            cv2.moveWindow('HSV Image', 10, 500)
            cv2.waitKey(1)
        return


    
    def RunWaitForPedestrianState(self):
        TOL_SKIN_X = 50
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
            stacked_contour = np.vstack(skin_cntrs)
            x,y,w,h = cv2.boundingRect(stacked_contour)
            cv2.rectangle(self.camera_feed,(x,y),(x+w,y+h),(0,255,0),2) 
            cv2.putText(self.camera_feed, 'Pedestrian', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for c in skin_cntrs:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
            if abs(cX- skin_mask.shape[1]//2) < TOL_SKIN_X:
                self.state = ControllerState.DRIVE_OUTER_LOOP
        if DEBUG_SKIN_MASK:
            cv2.imshow('Hazard Detection', self.downsample_image(skin_mask, 2))
            cv2.waitKey(1)



    def RunWaitForTruckState(self):
        TOL_TRUCK_X = 50    
        hsv_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2HSV)
        min_headlight = (0, 190, 30)
        max_headlight = (10, 210, 50)
        headlight_mask = cv2.inRange(hsv_feed, min_headlight, max_headlight)
        contours, hierarchy = cv2.findContours(headlight_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = [c for c in contours if cv2.contourArea(c) >= 2]
            if len(contours) > 0:
                moments = cv2.moments(max(contours, key=cv2.contourArea))
                cX = int(moments["m10"] / moments["m00"])
                if abs(cX - headlight_mask.shape[1]//3) < TOL_TRUCK_X:
                    self.state = ControllerState.DRIVE_INNER_LOOP
                    self.truck_passed = True
                    time.sleep(1) # wait 0.2 seconds for truck to pass
        cv2.imshow("Hazard Detection", self.downsample_image(headlight_mask, 2))
        return



# ==================== Utility Functions =======================
    def label_license_plate(self, image):
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
                        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),6) 
                        cv2.putText(image, 'License Plate', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, 'License Plate detected', (20, image.shape[0]-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 4)
                        cv2.imshow('License Plate', hsv_feed[y:y+h, x:x+w])
                        cv2.waitKey(1)





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
            cv2.imshow('Hazard Detection', self.downsample_image(red_mask, 2))
            cv2.moveWindow('Hazard Detection', 10, 500)
            cv2.waitKey(1)
        return False



    def call_outer_loop_driving_model(self, camera_feed):
        """
        Forward passes a cv2 camera image through the driving network and returns
        softmax probabalistic outputs.
        Ex: (Image of left turning road) ---> [0.2, 0.7, 0.1]
                                                F    L    R 
        """
        camera_feed_gray =  self.downsample_image(camera_feed, self.image_resize_factor, color_converter=cv2.COLOR_BGR2GRAY) # downsample
        camera_feed_gray = tf.expand_dims(camera_feed_gray, 0) # expand batch dim = 1
        softmaxes = tf.squeeze(self.outer_loop_driving_model(camera_feed_gray),0) # Squeeze output shape: (1, N) --> (N)
        return softmaxes
 


    def call_inner_loop_driving_model(self, camera_feed):
        """
        Forward passes a cv2 camera image through the driving network and returns
        softmax probabalistic outputs.
        Ex: (Image of left turning road) ---> [0.2, 0.7, 0.1]
                                                F    L    R 
        """
        camera_feed_gray =  self.downsample_image(camera_feed, self.image_resize_factor, color_converter=cv2.COLOR_BGR2GRAY) # downsample
        camera_feed_gray = tf.expand_dims(camera_feed_gray, 0) # expand batch dim = 1
        softmaxes = tf.squeeze(self.inner_loop_driving_model(camera_feed_gray),0) # Squeeze output shape: (1, N) --> (N)
        return softmaxes


    
    def convert_image_topic_to_cv_image(self, camera_topic):
        return self.bridge.imgmsg_to_cv2(camera_topic, 'bgr8')



    def convert_action_to_cmd_vel(self, action):
        """
        Maps an action (0, 1, or 2) to a Twist() object to publish to ROS network
        """
        if self.state == ControllerState.DRIVE_OUTER_LOOP:
            linspeed = self.outer_loop_linear_speed
            angspeed = self.outer_loop_angular_speed
        elif self.state == ControllerState.DRIVE_INNER_LOOP:
            linspeed = self.inner_loop_linear_speed
            angspeed = self.inner_loop_angular_speed
        else:
            print("Error: Not in a model driving state.")
        
        out = Twist()
        out.linear.x = linspeed 
        if action == 1:
            out.angular.z = angspeed
        elif action == 2:
            out.angular.z = -1 * angspeed
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
        # cv2.putText(img=out, text='Take Pictures: ' + str(self.take_pictures), org=(20, 180), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        return out