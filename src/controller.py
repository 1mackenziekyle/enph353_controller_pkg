from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

import rospy
import cv2
import numpy as np
import time




def step_robot(data):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
    shrink = 2
    img_shape = (cv_image.shape[1] // shrink, cv_image.shape[0] // shrink)
    cv2.imshow("Robot view", cv2.resize(cv_image, img_shape, interpolation=cv2.INTER_AREA))
    cv2.waitKey(1)
    move = Twist()
    move.angular.z = 0.5
    # pub.publish(move) 




# forever loop
rospy.init_node('controller', anonymous=True)
pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=step_robot)
# forever
rospy.spin()