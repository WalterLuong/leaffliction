from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import polars as pl
import pandas as pd
import cv2
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#from cloud_tpu_client import Client
#c = Client()
#c.configure_tpu_version(tf.__version__, restart_type="always")


IMG_SIZE = 224


dataset_path = os.listdir("data/images/Apple")

NUM_CLASSES = len(dataset_path)

print(dataset_path)

class_labels = []

for item in dataset_path:
    # Get all the file names
    all_classes = os.listdir('data/images/Apple' + '/' + item)
    # print(all_classes)

    # Add them to the list
    for room in all_classes:
        class_labels.append(
            (item, str('data/images/Apple' + '/' + item) + '/' + room))
        # print(class_labels[:5])

# Build a dataframe
df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
# print(df.head())
# print(df.tail())

# Let's check how many samples for each category are present
print("Total number of images in the dataset: ", len(df))

label_count = df['Labels'].value_counts()
print(label_count)

images = []
labels = []

path = 'data/images/Apple/'
for i in dataset_path:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(i)

images = np.array(images).astype('float32') / 255.0

y = LabelEncoder().fit_transform(df['Labels'].values)

y = y.reshape(-1, 1)
ct = ColumnTransformer(
    [('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y).toarray()

images, Y = shuffle(images, Y, random_state=1)


train_x, test_x, train_y, test_y = train_test_split(
    images, Y, test_size=0.2, random_state=415)

# inspect the shape of the training and testing.
#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)
#print(test_y.shape)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()


NUM_CLASSES = len(dataset_path)


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("accuracy.png")
    #plt.show()


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(
        include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


with strategy.scope():
    model = build_model(num_classes=NUM_CLASSES)

epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_x, train_y, epochs=epochs, verbose=2)
plot_hist(hist)


# def unfreeze_model(model):
#    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
#    for layer in model.layers[-20:]:
#        if not isinstance(layer, layers.BatchNormalization):
#            layer.trainable = True

#    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#    model.compile(
#        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
#    )


# unfreeze_model(model)

# epochs = 10  # @param {type: "slider", min:8, max:50}
# hist = model.fit(train_x, train_y, epochs=epochs, verbose=2)
# plot_hist(hist)

# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
