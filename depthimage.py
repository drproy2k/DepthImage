##############################################################################################################
# disparityimage.py
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
# Note : Depth = Bf / disparity

import numpy as np
import cv2
from matplotlib import pyplot as plt

# def disparity(self):
#     matcher = cv2.StereoBM_create(1024, 7)
#     disparity = matcher.compute(cv2.cvtColor(self.images[0], cv2.COLOR_BGR2GRAY),
#                                 cv2.cvtColor(self.images[1], cv2.COLOR_BGR2GRAY))
#     self.process_output(disparity)


imgR = cv2.imread('Yeuna9x.png',0)
imgL = cv2.imread('SuXT483.png',0)

#stereo = cv2.StereoBM(1, 16, 15)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity,'gray')
plt.show()
