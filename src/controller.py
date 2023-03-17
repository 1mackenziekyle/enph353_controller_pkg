from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

import rospy
import cv2
import numpy as np
import time


class Controller:
    def __init__(self):
        self.move = Twist()
        self.frame = cv2.Mat(np.zeros((10,10)))

    def step(self, data):
        # shrink image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
        shrink = 2
        img_shape = (cv_image.shape[1] // shrink, cv_image.shape[0] // shrink)
        img_resize = cv2.resize(cv_image, img_shape, interpolation=cv2.INTER_AREA)
        # detect edges
        edges = cv2.Canny(img_resize, 80,100)
        # record move
        move_arr = [self.move.linear.x, self.move.linear.y, self.move.linear.z, self.move.angular.z]
        move_strs = ['x', 'y', 'z', 'angular']
        disp_move = [s + ': ' + str(m) for s, m in zip(move_strs, move_arr)]


        # display
        for i in range(len(disp_move)):
            cv2.putText(img_resize, disp_move[i], (20,20+40*i), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
        # cv2.imshow("Robot view - Edge Detection", edges)
        # cv2.waitKey(1)
        cv2.imshow("Robot view", img_resize)
        cv2.waitKey(1)
        # pub.publish(move) 

        self.frame = img_resize

    def record_move(self, data):
        self.move = data

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



# forever loop
rospy.init_node('controller', anonymous=True)
# initialize controller object
controller = Controller()

# publish move
# pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

# subscribers
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=controller.step)
rospy.Subscriber('R1/cmd_vel', Twist, callback=controller.record_move)
# forever
rospy.spin()