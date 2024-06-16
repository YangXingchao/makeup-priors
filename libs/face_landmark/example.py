import cv2
import sys

from libs.face_landmark import landmark

if __name__ == '__main__':

    img = cv2.imread("../../datasets/source/fit/gt/00000.jpg")
    a = landmark.detect_landmark(img)
    print(a)