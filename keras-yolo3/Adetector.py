# Importing all necessary libraries
import cv2
import os
from PIL import Image
# import csv module
import csv

# from Subscriber.detector import *

from yolo import YOLO, detect_video
from carType_Classifier import CarType
from CarColour_Classifier import ColourDetector

# Read the video from specified path
cam = cv2.VideoCapture("C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\video.mp4")
cam.set(cv2.CAP_PROP_FPS, 30.0)
fps = cam.get(cv2.CAP_PROP_FPS)
# print("fps:", fps, type(fps))

try:
    # creating a folder named data
    if not os.path.exists('../data'):
        os.makedirs('../data')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
yolo_obj = YOLO()
cartype = CarType()


def ROI(yolo_obj, frame, frame_n):
    img_n = 0
    image, BoundingList, timer = yolo_obj.detect_image(frame)
    car_list = list()
    if len(BoundingList) == 0:
        write_excel(frame_n, car_list)
    for Bounding_box in BoundingList:
        if Bounding_box['class'] == 'car':
            print("Car detected")
            top = Bounding_box['top']
            bottom = Bounding_box['bottom']
            left = Bounding_box['left']
            right = Bounding_box['right']

            image = image.crop((left + 1, top + 1, right + 1, bottom + 1))
            image.save("C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\data\\image_" +
                       str(frame_n) + "_" + str(img_n) + ".jpeg", 'JPEG')
            print("Created image: ", str(frame_n) + "_" + str(img_n))
            type = cartype.type_classifier(
                "C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\data\\image_" +
                str(frame_n) + "_" + str(img_n) + ".jpeg")
            car_list.append(type)

            # increasing counter so that it will
            # show how many frames are created
            img_n += 1
        else:
            print("Class: ", Bounding_box['class'])
    if frame_n == 0:
        write_excel(frame_n, car_list, False)
    else:
        write_excel(frame_n, car_list)

    # print("\nColour classifier") frame
    # ColourDetector("C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\data\\image_" + str(currentframe) + ".jpeg")


def write_excel(frame_num, car_info, write_flag=True):
    with open('CarClassifier.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        sedan = 0
        hatchback = 0
        if not write_flag:
            writer.writerow(["Frame No", "Sedan", "Hatchback", "Total Cars"])
            write_flag = True

        if len(car_info) > 0:
            for i in range(len(car_info)):
                if car_info[i] == "Sedan":
                    sedan += 1
                elif car_info[i] == "Hatchback":
                    hatchback += 1
            writer.writerow([frame_num, sedan, hatchback, len(car_info)])
        else:
            writer.writerow([frame_num, sedan, hatchback, len(car_info)])


while True:
    # reading from frame
    ret, frame = cam.read()
    if ret:
        frame = Image.fromarray(frame, 'RGB')
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print('\n****************\nCreating...' + name)
        # region(frame)
        ROI(yolo_obj, frame, currentframe)
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
