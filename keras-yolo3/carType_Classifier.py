import keras
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils, MobileNetV2
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.mobilenet import preprocess_input


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
        # print("Init")
        self.model, self.train_generator, self.validation_generator = defmodel()
        self.fit()

    def fit(self):
        # model, train_generator = defmodel()

        step_size_train = self.train_generator.n // self.train_generator.batch_size
        step_size_val = self.validation_generator.samples // self.validation_generator.batch_size
        self.model.fit_generator(generator=self.train_generator,
                                 steps_per_epoch=step_size_train,
                                 validation_data=self.validation_generator,
                                 validation_steps=step_size_val,
                                 # callbacks=callbacks_list,
                                 epochs=20)

        tf.saved_model(self.model, 'saved_model')
        # step_size_train = self.train_generator.n // self.train_generator.batch_size
        # self.model.fit_generator(generator=self.train_generator,
        #                         steps_per_epoch=step_size_train,
        #                         epochs=2)

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(180, 180))
        img_tensor = image.img_to_array(img)  # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the
        # model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.  # imshow expects values in the range [0, 1]

        return img_tensor

    def type_classifier(self, image_path):
        model = keras.models.load_model('saved_model')
        new_image = self.load_image(image_path)
        pred = model.predict(new_image)

        rounded = float(np.round(pred))
        print("\nPrediction: ", pred, rounded)
        if rounded == 0.0:
            return "Hatchback"
        elif rounded == 1.0:
            return "Sedan"

        # p1 = [labels[np.argmax(pred)]]  # retrieving the max of the output
        # print(pred, "\n", p1)
        # Y_test1 = [labels[np.argmax(Y_test[i, :])] for i in range(Y_test.shape[0])]
