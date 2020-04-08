import numpy as np
import cv2


def ColourDetector(image):
    Cols = []
    Pred = {'Red': 0, 'Blue': 0, 'Black': 0, 'White': 0, 'Silver': 0}
    new_image1 = cv2.imread(image)
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),  # Red
        ([86, 31, 4], [220, 88, 50]),  # Blue
        ([0, 0, 0], [0, 0, 0]),  # Black
        ([0, 0, 255], [255, 255, 255]),  # white
        ([0, 0, 192], [192, 192, 192]),  # silver
    ]
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(new_image1, lower, upper)
        output = cv2.bitwise_and(new_image1, new_image1, mask=mask)
        Colour = np.sum(output)  # sum of all pixels
        Cols.append(Colour)
    col = np.argmax(Cols)  # Returns index of max element
    if col == 0:
        Pred['Red'] += 1
    if col == 1:
        Pred['Blue'] += 1
    if col == 2:
        Pred['Black'] += 1
    if col == 3:
        Pred['White'] += 1
    if col == 4:
        Pred['Silver'] += 1
    print("Colour: ", Pred)
