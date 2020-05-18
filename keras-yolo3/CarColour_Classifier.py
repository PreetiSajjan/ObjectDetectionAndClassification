import cv2
import numpy as np
from Results import Result
import time

# img_dir = "C:\\Users\\HP\\PycharmProjects\\MyProject\\ASSSSGGGGGG2\\keras-yolo3\\data" # Enter Directory of all images
# data_path = os.path.join(img_dir,'*g')
# files = glob.glob(data_path)
res = Result()

def ColourDetector(img_path, frame, frame_n, type, max_car, bound):
    new_image1 = cv2.imread(img_path)
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),  # Red
        ([86, 31, 4], [220, 88, 50]),  # Blue
        ([0, 0, 0], [180, 255, 30]),  # Black
        ([0, 0, 200], [180, 20, 255]),  # white
        ([0, 0, 192], [192, 192, 192]),  # silver
    ]
    Cols = []
    DetectedColour = ()

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(new_image1, lower, upper)
        output = cv2.bitwise_and(new_image1, new_image1, mask=mask)
        Colour = np.sum(output)  # sum of all pixels
        Cols.append(Colour)
    col = np.argmax(Cols)  # Returns index of max element

    if col == 0:
        DetectedColour = 'Red'
    if col == 1:
        DetectedColour = 'Blue'
    if col == 2:
        DetectedColour = 'Black'
    if col == 3:
        DetectedColour = 'White'
    if col == 4:
        DetectedColour = 'Silver'

    ColEnd = time.time()
    res.update_result(frame, frame_n, type, DetectedColour,  max_car, bound)
    return ColEnd
