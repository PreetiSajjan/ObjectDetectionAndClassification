import keras
import numpy as np
from keras.applications import MobileNetV2
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import time

from CarColour_Classifier import *

def defmodel():
    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(180, 180, 3))  # imports the mobilenet model and discards the last 1000 neuron layer.
    base_model.trainable = False

    model = Sequential([
        base_model,
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    # opt = RMSProp(lr=learning_rate)
    model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                       validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\dataset',
        target_size=(180, 180),
        #colour_mode = 'rgb',
        batch_size=64,
        class_mode='binary',
        subset='training',
        shuffle=True)

    validation_generator = train_datagen.flow_from_directory(
        'C:\\Users\\User\\PycharmProjects\\RTAIAssignment2\\keras-yolo3\\dataset',
        # same directory as training data
        target_size=(180, 180),
        #colour_mode='rgb',
        batch_size=64,
        class_mode='binary',
        subset='validation',
        shuffle=True)  # set as validation data

    return model, train_generator, validation_generator


class CarType:
    def __init__(self):
        # self.model, self.train_generator, self.validation_generator = defmodel()
        # self.fit()
        self.model_saved = keras.models.load_model('saved_model')

    def fit(self):
        # model, train_generator = defmodel()

        step_size_train = self.train_generator.n // self.train_generator.batch_size
        step_size_val = self.validation_generator.samples // self.validation_generator.batch_size
        self.model.fit_generator(generator=self.train_generator,
                                 steps_per_epoch=step_size_train,
                                 validation_data=self.validation_generator,
                                 validation_steps=step_size_val,
                                 # callbacks=callbacks_list,
                                 epochs=10)
        self.model.save('saved_model')


    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(180, 180))
        img_tensor = image.img_to_array(img)  # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the
        # model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.  # imshow expects values in the range [0, 1]

        return img_tensor

    def type_classifier(self, image_path, frame, frame_n, max_car, bound):
        new_image = self.load_image(image_path)
        pred = self.model_saved.predict(new_image)

        rounded = float(np.round(pred))

        # print("\nPrediction: ", pred, rounded)
        if rounded == 0.0:
            type = "Hatchback"
        elif rounded == 1.0:
            type = "Sedan"

        TypeEnd = time.time()

        ColEnd = ColourDetector(image_path, frame, frame_n, type, max_car, bound)
        return TypeEnd, ColEnd
