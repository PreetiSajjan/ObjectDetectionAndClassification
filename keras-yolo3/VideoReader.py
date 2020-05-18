import os
import cv2
from PIL import Image
from ObjectDetector import Detector
from Statistics import *

# frame
currentframe = 1
object_detector = Detector()

# Read the video from specified path
cam = cv2.VideoCapture("C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\video.mp4")
#cam = cv2.VideoCapture("C:\\Users\\User\\Desktop\\Experiment\\VIRAT.mp4")
cam.set(cv2.CAP_PROP_FPS, 30)
fps = cam.get(cv2.CAP_PROP_FPS)

try:
    # creating a folder named data
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if not os.path.exists('../ClassifiedData'):
        os.makedirs('../ClassifiedData')
    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

while True:
    # reading from frame
    ret, frame = cam.read()
    if ret:
        #print(frame.shape)
        frame = Image.fromarray(frame, 'RGB')

        # video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print('\n****************\nCreating...' + name)
        # region(frame)
        object_detector.ROI(frame, currentframe)
        currentframe += 1
    else:
        print("Winding up!!")
        calculate_stats()
        object_detector.release(currentframe)
        print("Exit")
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()