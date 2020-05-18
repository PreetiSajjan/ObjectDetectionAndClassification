import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

def calculate_stats():
    Ground_Truth = pd.read_excel("C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\Ground Truth.xlsx")
    Classifier = pd.read_excel("C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\CarClassifier.xlsx")
    TotalCarsTrue = Ground_Truth['Total Cars']
    TotalCarsTrue = TotalCarsTrue.iloc[1:]
    TotalCarsPredict = Classifier['Total Cars']
    TotalCarsPredict = TotalCarsPredict.iloc[1:]

    ## Ground Truth
    # Total Sedan Cars
    TotalSedanTrue_temp = Ground_Truth.drop(
        ['Frame No.', 'Hatchback', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Total Cars'], axis=1)
    TrueSedan_Black = np.where(TotalSedanTrue_temp['Sedan'] == 1, 1, 0)
    TrueSedan_Silver = np.where(TotalSedanTrue_temp['Unnamed: 2'] == 1, 1, 0)
    TrueSedan_Red = np.where(TotalSedanTrue_temp['Unnamed: 3'] == 1, 1, 0)
    TrueSedan_White = np.where(TotalSedanTrue_temp['Unnamed: 4'] == 1, 1, 0)
    TrueSedan_Blue = np.where(TotalSedanTrue_temp['Unnamed: 5'] == 1, 1, 0)
    TrueSedan = TrueSedan_Black + TrueSedan_Silver + TrueSedan_Red + TrueSedan_White + TrueSedan_Blue

    # Total Hatchback Cars
    TotalHatchbackTrue_temp = Ground_Truth.drop(
        ['Frame No.', 'Sedan', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Total Cars'], axis=1)
    TrueHatchback_Black = np.where(TotalHatchbackTrue_temp['Hatchback'] == 1, 1, 0)
    TrueHatchback_Silver = np.where(TotalHatchbackTrue_temp['Unnamed: 7'] == 1, 1, 0)
    TrueHatchback_Red = np.where(TotalHatchbackTrue_temp['Unnamed: 8'] == 1, 1, 0)
    TrueHatchback_White = np.where(TotalHatchbackTrue_temp['Unnamed: 9'] == 1, 1, 0)
    TrueHatchback_Blue = np.where(TotalHatchbackTrue_temp['Unnamed: 10'] == 1, 1, 0)
    TrueHatchback = TrueHatchback_Black + TrueHatchback_Silver + TrueHatchback_Red + TrueHatchback_White + TrueHatchback_Blue

    ## Classifier
    # Total Sedan Cars

    TotalSedanPred_temp = Classifier.drop(
        ['Frame No.', 'Hatchback', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Total Cars'], axis=1)
    PredSedan_Black = np.where(TotalSedanPred_temp['Sedan'] == 1, 1, 0)
    PredSedan_Silver = np.where(TotalSedanPred_temp['Unnamed: 2'] == 1, 1, 0)
    PredSedan_Red = np.where(TotalSedanPred_temp['Unnamed: 3'] == 1, 1, 0)
    PredSedan_White = np.where(TotalSedanPred_temp['Unnamed: 4'] == 1, 1, 0)
    PredSedan_Blue = np.where(TotalSedanPred_temp['Unnamed: 5'] == 1, 1, 0)
    PredSedan = PredSedan_Black + PredSedan_Silver + PredSedan_Red + PredSedan_White + PredSedan_Blue

    # Total Hatchback Cars
    TotalHatchbackPred_temp = Classifier.drop(
        ['Frame No.', 'Sedan', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Total Cars'], axis=1)
    PredHatchback_Black = np.where(TotalHatchbackPred_temp['Hatchback'] == 1, 1, 0)
    PredHatchback_Silver = np.where(TotalHatchbackPred_temp['Unnamed: 7'] == 1, 1, 0)
    PredHatchback_Red = np.where(TotalHatchbackPred_temp['Unnamed: 8'] == 1, 1, 0)
    PredHatchback_White = np.where(TotalHatchbackPred_temp['Unnamed: 9'] == 1, 1, 0)
    PredHatchback_Blue = np.where(TotalHatchbackPred_temp['Unnamed: 10'] == 1, 1, 0)
    PredHatchback = PredHatchback_Black + PredHatchback_Silver + PredHatchback_Red + PredHatchback_White + PredHatchback_Blue

    # Query-1
    TotalCarsScore = f1_score(TotalCarsTrue, TotalCarsPredict, average='weighted')
    print("\nCalculating F1 score:\n")
    print("F1 Score for Total Cars in each frame: ", TotalCarsScore)
    # Query-2
    Score_Sedan = f1_score(TrueSedan, PredSedan, average='weighted')
    print("F1 Score for Sedan Cars in each frame: ", Score_Sedan)
    Score_Hatchback = f1_score(TrueHatchback, PredHatchback, average='weighted')
    print("F1 Score for Hatchback Cars in each frame: ", Score_Hatchback)
    # Query-3
    ## SEDAN
    print("SEDAN CARS")
    Black_Sedan = f1_score(TrueSedan_Black, PredSedan_Black, average='weighted')
    print("F1 Score for Sedan Cars Black Colour: ", Black_Sedan)
    Silver_Sedan = f1_score(TrueSedan_Silver, PredSedan_Silver, average='weighted')
    print("F1 Score for Sedan Cars Silver Colour: ", Silver_Sedan)
    Red_Sedan = f1_score(TrueSedan_Red, PredSedan_Red, average='weighted')
    print("F1 Score for Sedan Cars Red Colour: ", Red_Sedan)
    White_Sedan = f1_score(TrueSedan_White, PredSedan_White, average='weighted')
    print("F1 Score for Sedan Cars White Colour: ", White_Sedan)
    Blue_Sedan = f1_score(TrueSedan_Blue, PredSedan_Blue, average='weighted')
    print("F1 Score for Sedan Cars Blue Colour: ", Blue_Sedan)
    ## HATCHBACK
    print("\nHATCHBACK CARS")
    Black_Hatchback = f1_score(TrueHatchback_Black, PredHatchback_Black, average='weighted')
    print("F1 Score for Hatchback Cars Black Colour: ", Black_Hatchback)
    Silver_Hatchback = f1_score(TrueHatchback_Silver, PredHatchback_Silver, average='weighted')
    print("F1 Score for Hatchback Cars Silver Colour: ", Silver_Hatchback)
    Red_Hatchback = f1_score(TrueHatchback_Red, PredHatchback_Red, average='weighted')
    print("F1 Score for Hatchback Cars Red Colour: ", Red_Hatchback)
    White_Hatchback = f1_score(TrueHatchback_White, PredHatchback_White, average='weighted')
    print("F1 Score for Hatchback Cars White Colour: ", White_Hatchback)
    Blue_Hatchback = f1_score(TrueHatchback_Blue, PredHatchback_Blue, average='weighted')
    print("F1 Score for Hatchback Cars Blue Colour: ", Blue_Hatchback)
