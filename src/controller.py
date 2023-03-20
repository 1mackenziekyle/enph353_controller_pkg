from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

import rospy
import cv2
import numpy as np
import datetime
from keras import models, layers
from enum import Enum
import tensorflow as tf

# Driving Mode
class Driving_Mode(Enum):
    MANUAL = 1,
    MODEL = 2
    TAKE_PICTURES = 3

class Controller:
    def __init__(self, driving_mode, image_save_location, start_snapshots, snapshot_freq, 
                 image_resize_factor, publisher, model_path, linear_speed, angular_speed):
        self.bridge = CvBridge()
        self.iters = 0
        self.vels = Twist()
        self.driving_mode = driving_mode
        self.image_save_location=image_save_location
        self.start_snapshots = start_snapshots
        self.snapshot_freq = snapshot_freq
        self.image_resize_factor = image_resize_factor
        self.publisher=publisher
        self.model_path = model_path
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.model=None
        

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def step(self, data):
        self.iters+=1
        # read camera message
        raw_camera = self.read_camera_message(data)
        # show video output
        self.show_camera_feed(raw_camera)
        # save picture
        if self.driving_mode == Driving_Mode.TAKE_PICTURES and self.iters > self.start_snapshots and self.iters % self.snapshot_freq == 0:
            self.save_image(cv2.cvtColor(self.resize(raw_camera, self.image_resize_factor), cv2.COLOR_BGR2GRAY), str([self.vels.linear.x, self.vels.angular.z]) + str(datetime.datetime.now()))
        # model
        elif self.driving_mode == Driving_Mode.MODEL:
            action = self.choose_action(raw_camera)
            self.publish_cmd_vel(action)

    def choose_action(self,image):
        image = self.resize(image, self.image_resize_factor)
        return np.argmax(self.model(tf.expand_dims(image,0)))
    
    def publish_cmd_vel(self, action):
        out = Twist()
        if action == 0:
            out.linear.x = 0.5 * self.linear_speed
        elif action == 1:
            out.angular.z = 1.0 * self.angular_speed
        else:
            out.angular.z = -1.0 * self.angular_speed
        self.publisher.publish(out)

    def show_camera_feed(self, raw_camera_image):
        cv2.imshow("Robot view", self.display_velocities(self.resize(raw_camera_image, 2)))
        cv2.waitKey(1)

    def read_camera_message(self, msg):
        return self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def read_velocities(self, data):
        self.vels = data

    def save_image(self, image: cv2.Mat, filename: str):
        print(' saving image', self.image_save_location + '/' + filename + '.jpg')
        try:
            cv2.imwrite(self.image_save_location + '/' + filename + '.png', image)
        except:
            print("failed saving image.")

    def draw_lines(edge_img, out):
        lines = cv2.HoughLines(edge_img, 1, np.pi / 180, 150, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(out, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
        return out

    def resize(self, img, factor):
        shape = img.shape
        resized_shape = (shape[1] // factor, shape[0] // factor)
        return cv2.resize(img, resized_shape, interpolation=cv2.INTER_AREA)
    
    def get_edges_image(self, image, min, max):
        return cv2.Canny(image, min, max)
    
    def display_velocities(self, input_image):
        out = input_image.copy()
        velocity_array = [self.vels.linear.x, self.vels.linear.y, self.vels.linear.z, self.vels.angular.z]
        velocity_attributes_and_values = [s + ': ' + str(m) for s, m in zip(['x', 'y', 'z', 'angular'], velocity_array)]
        for i in range(len(velocity_attributes_and_values)):
            cv2.putText(img=out, text=velocity_attributes_and_values[i], org=(20,20+40*i), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(img=out, text="iters: " + str(self.iters), org=(100, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255,255,255), thickness=2)
        return out





# ============ Start Loop =============
# rospy.init_node('controller', anonymous=True)

# initialize controller object
# controller = Controller(
#     driving_mode=Driving_Mode.TAKE_PICTURES,
#     image_save_location='./assets/images/outer_lap2',
#     start_snapshots=100,
#     snapshot_freq=1,
#     image_resize_factor=20,
#     publisher=rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1),
#     model_path='./assets/models/drive_outer_loop/',
#     linear_speed=1.0,
#     angular_speed=1.0
#     )
# if controller.driving_mode is Driving_Mode.MODEL:
#     controller.load_model()


# # subscribers
# rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=controller.step)
# rospy.Subscriber('R1/cmd_vel', Twist, callback=controller.read_velocities)
# # forever
# rospy.spin()