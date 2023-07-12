import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# GPU memory management
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Plant selection and dataset paths
plant = 'Apple'
dataset_train_path = Path('data', 'images', plant)
dataset_valid_path = Path('data', 'valid', plant)


# Load dataset
traindata = ImageDataGenerator().flow_from_directory(
    directory=dataset_train_path, target_size=(200, 200))
testdata = ImageDataGenerator().flow_from_directory(
    directory=dataset_valid_path, target_size=(200, 200))


# Model definition (VGG16)
input_shape = (200, 200, 3)
inputs = Input(shape=input_shape)

x = Conv2D(filters=64, kernel_size=(3, 3),
           padding="same", activation="relu")(inputs)
x = Conv2D(filters=64, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=128, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = Conv2D(filters=128, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=256, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = Conv2D(filters=256, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = Conv2D(filters=256, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=512, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(filters=512, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3, 3),
           padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Flatten()(x)
x = Dense(units=4096, activation="relu")(x)
x = Dense(units=4096, activation="relu")(x)
x = Dense(units=1000, activation="relu")(x)
outputs = Dense(units=8, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt,
              loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()


# Train
checkpoint = ModelCheckpoint(f'{plant}_vgg16.h5', monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max', save_freq=1)
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=20, verbose=1, mode='max')
callbacks = [checkpoint, early_stopping]

hist = model.fit(traindata, batch_size=2, steps_per_epoch=10, epochs=100,
                 verbose=1, callbacks=callbacks)


# Plot
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('loss.png')
plt.show()
