# Importing all necessary libraries
# %matplotlib inline
import matplotlib.pyplot as plt
import time

from CarType_Classifier import CarType
# from CarColour_Classifier import *
# from Statistics import *
from Results import Result
from yolo import YOLO

result = Result()


class Detector:
    def __init__(self):
        self.yolo_obj = YOLO()
        self.cartype = CarType()
        self.ROItimer = list()
        self.ROITypeTimer = list()
        self.ROITypeColourTimer = list()

    def ROI(self, frame, frame_n):
        img_n = 0
        SumTypeEnd, SumColEnd = 0, 0
        start = time.time()
        image, BoundingList = self.yolo_obj.detect_image(frame)
        ROIEnd = time.time()
        self.ROItimer.append(ROIEnd - start)

        if len(BoundingList) == 0:
            result.update_result(frame, frame_n, 0, 0, 0, 0)

        else:
            l = [a['class'] for a in BoundingList]
            max_car = l.count('car')

            for Bounding_box in BoundingList:
                bound = list()
                if Bounding_box['class'] == 'car':
                    print("Car detected")
                    top = Bounding_box['top']
                    bottom = Bounding_box['bottom']
                    left = Bounding_box['left']
                    right = Bounding_box['right']
                    bound.append(left)
                    bound.append(top)
                    image = image.crop((left + 1, top + 1, right + 1, bottom + 1))
                    image.save("C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\data\\image_" +
                               str(frame_n) + "_" + str(img_n) + ".jpeg", 'JPEG')

                    # print("Created image: ", str(frame_n) + "_" + str(img_n))
                    path = "C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\" \
                           "keras-yolo3\\data\\image_" + str(frame_n) + "_" + str(img_n) + ".jpeg"

                    TypeEnd, ColEnd = self.cartype.type_classifier(path, frame, frame_n, max_car, bound)
                    SumTypeEnd += TypeEnd
                    SumColEnd += ColEnd
                    img_n += 1
                    max_car -= 1

            if img_n == 0:
                result.update_result(frame, frame_n, 0, 0, 0, 0)

        self.ROITypeTimer.append(SumTypeEnd - start)
        self.ROITypeColourTimer.append(SumColEnd - start)

    def release(self, frame_n):
        result.show_video(0, 0, True)
        numbers = range(frame_n - 1)
        plt.plot(numbers, self.ROItimer, 'r--', alpha=0.7)
        plt.plot(numbers, self.ROITypeTimer, 'b--', alpha=0.5)
        plt.plot(numbers, self.ROITypeColourTimer, 'y--', alpha=0.5)
        plt.xlabel("Frames")
        plt.ylabel("Time (sec)")
        plt.title("Execution time taken per Frame")
        plt.legend(['ROI Timer', 'ROI+Type Timer', 'ROI+Type+Colour Timer'], loc='upper right')
        plt.show()
