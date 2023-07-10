import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset_train_path = Path('data', 'images', 'Apple')
dataset_valid_path = Path('data', 'valid', 'Apple')


traindata = ImageDataGenerator().flow_from_directory(
    directory=dataset_train_path, target_size=(200, 200))
testdata = ImageDataGenerator().flow_from_directory(
    directory=dataset_valid_path, target_size=(200, 200))

model = Sequential()
model.add(Conv2D(input_shape=(200, 200, 3), filters=64,
          kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
          padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4, activation="softmax"))

opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt,
              loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='accuracy', verbose=1,
                             save_best_only=True, mode='max', save_freq=1)
early_stopping = EarlyStopping(monitor='accuracy',
                               patience=20, verbose=1, mode='max')
callbacks = [checkpoint, early_stopping]

hist = model.fit(traindata, batch_size=2, steps_per_epoch=10, epochs=100,
                 verbose=1, callbacks=callbacks)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
# plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.savefig('vgg16_1.png')
plt.show()
