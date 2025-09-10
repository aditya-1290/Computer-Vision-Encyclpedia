"""
Optical Flow: Estimate motion between consecutive frames.

Implementation uses OpenCV for Lucas-Kanade optical flow.

Theory:
- Optical flow: Apparent motion of objects in visual field.
- Assumption: Brightness constancy, small motions.

Math: I(x,y,t) = I(x+dx,y+dy,t+dt), solve for dx, dy.

Reference:
- Lucas & Kanade, An Iterative Image Registration Technique with an Application to Stereo Vision, 1981
"""

import cv2
import numpy as np

def compute_optical_flow(frame1, frame2):
    """
    Compute optical flow using Lucas-Kanade method.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

if __name__ == "__main__":
    # Assume two frames
    # frame1 = cv2.imread("frame1.jpg")
    # frame2 = cv2.imread("frame2.jpg")
    # flow = compute_optical_flow(frame1, frame2)
    # print(f"Flow shape: {flow.shape}")
    print("Optical flow function defined.")
