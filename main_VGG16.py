import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.applications import VGG16
from keras.src.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
import time

# FILEPATH = 'C:\Users\ishan\PycharmProjects\skin_cancer_identification'
FILEPATH = 'E:\Projects\skin_cancer_identification'

metadata = pd.read_csv('%s\dataset\HAM10000_metadata.csv' % FILEPATH)

label_arr = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
label_dict = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

# training config:
epochs = 32
batch_size = 64
# re-size all the images to this
image_size = 128

input_shape = (128, 128, 3)
num_classes = 7

def preprocess():
    # Preprocess the data

    X = []
    y = []

    for index, row in metadata.iterrows():
        if index % 1000 == 0:
            print(f"Processing image {index}")
        img_id = row['image_id'] + '.jpg'
        img_path1 = os.path.join('%s\dataset\skin-cancer-dataset\Skin Cancer' % FILEPATH, img_id)
        if os.path.exists(img_path1):
            img_path = img_path1
        else:
            print(f"Image file does not exist: {img_id}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
        y.append(row['dx'])

    X = np.array(X)
    X = X / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train():
    x_test, x_train, y_test, y_train = preprocess()

    y_train = [label_dict[label] for label in y_train]
    y_test = [label_dict[label] for label in y_test]

    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    # Define the data augmentation transformations
    datagen = ImageDataGenerator(rotation_range=30,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 preprocessing_function=keras.applications.resnet50.preprocess_input)

    # Create the training generator
    train_generator = datagen.flow(x_train, y_train, batch_size=32)

    # Create the validation generator
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(x_test, y_test, batch_size=32)

    # Create the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    opt = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Apply data augmentation to the training set
    train_generator = datagen.flow(x_train, y_train, batch_size=32)

    history_fit = model.fit(train_generator, steps_per_epoch=len(x_train) // 32,
                        validation_data=val_generator, validation_steps=len(x_test) // 32, epochs=50)

    # Plot the accuracy and loss curves
    plt.plot(history_fit.history['accuracy'], label='Training accuracy')
    plt.plot(history_fit.history['val_accuracy'], label='Validation accuracy')
    plt.legend()
    # plt.show()
    plt.savefig('VGG16_accuracy.png')

    plt.plot(history_fit.history['loss'], label='Training loss')
    plt.plot(history_fit.history['val_loss'], label='Validation loss')
    plt.legend()
    # plt.show()
    plt.savefig('VGG16_loss.png')

    # Save the model
    model.save('SkinCancer_CNN_VGG16.h5')

    # print("Loss of the VGG16 model is - ", history_fit.evaluate(x_test, y_test)[0])
    # print("Accuracy of the VGG16 model is - ", history_fit.evaluate(x_test, y_test)[1] * 100, "%")



def evaluator():
    model = keras.models.load_model('%s\SkinCancer_CNN_VGG16.h5' % FILEPATH)
    x_test, x_train, y_test, y_train = preprocess()

    # Check out the layers in our model
    model.summary()

    print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
    print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")


def predictor(file_name):
    model = keras.models.load_model('%s\SkinCancer_CNN_VGG16.h5' % FILEPATH)

    img_path1 = os.path.join('%s\dataset\skin-cancer-dataset\Skin Cancer' % FILEPATH,
                             file_name)
    if os.path.exists(img_path1):
        img_path = img_path1
    else:
        print(f"Image file does not exist:")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")

    img = cv2.resize(img, (image_size, image_size))
    plt.imshow(img)
    img = img / 255.0
    pred = model.predict(np.array([img]))
    print(pred)
    res = [round(x, 3) for x in pred[0]]
    # print(res)
    final_res ={}
    for idx, x in enumerate(res):
        final_res.update({label_arr[idx]: x})
    print(final_res)


def main():
    start_time = time.time()
    # train()
    # evaluator()
    predictor("ISIC_0027850.jpg")
    print("--- %s VGG16 - seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
