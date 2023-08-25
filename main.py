import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

FILEPATH = 'E:\Projects\skin_cancer_identification'

metadata = pd.read_csv('%s\dataset\HAM10000_metadata.csv' % FILEPATH)

label_arr = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
label_dict = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

def preprocess():
    # Preprocess the data
    image_size = 32
    X = []
    y = []
    for index, row in metadata.iterrows():
        if index % 1000 == 0:
            print(f"Processing image {index}")
        img_id = row['image_id'] + '.jpg'
        img_path1 = os.path.join('%s\dataset\skin-cancer-dataset\Skin Cancer' % FILEPATH,
                                 img_id)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert labels from string to numerical categories
    y_train = [label_dict[label] for label in y_train]
    y_test = [label_dict[label] for label in y_test]
    # Convert the labels to one-hot encoded vectors
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    return X_test, X_train, y_test, y_train


def train():
    x_test, x_train, y_test, y_train = preprocess()

    # Define the model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # Compile the model
    # opt = keras.optimizersAdam(lr=0.01)
    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=32, batch_size=64)

    # Plot the accuracy and loss curves
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

    print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
    print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")

    # Save the model
    model.save('SkinCancer_CNN.h5')


def evaluator():
    model = keras.models.load_model('%s\SkinCancer_CNN.h5' % FILEPATH)
    x_test, x_train, y_test, y_train = preprocess()

    # Check out the layers in our model
    model.summary()

    print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
    print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")


def predictor(file_name):
    model = keras.models.load_model('%s\SkinCancer_CNN.h5' % FILEPATH)
    image_size = 32

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


if __name__ == '__main__':
    train()
    # evaluator()
    # predictor("ISIC_0027850.jpg")
