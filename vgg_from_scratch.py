import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] \
        in %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# GPU memory management
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Plant selection and dataset paths
plant = 'All'
dataset_train_path = Path('training')
dataset_valid_path = Path('validation')


# Load dataset
traindata = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=dataset_train_path, target_size=(200, 200))
testdata = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=dataset_valid_path, target_size=(200, 200))
num_classes = len(traindata.class_indices)


# Model definition (VGG16)
model = models.Sequential()
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 1),
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(8, activation="softmax"))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

model.summary()


# Train
checkpoint = ModelCheckpoint(
    f'{plant}_vgg16.h5',
    monitor='accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_freq=1
)
early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=20,
    verbose=1,
    mode='max'
)
callbacks = [checkpoint, early_stopping]

hist = model.fit(
    traindata,
    batch_size=2,
    steps_per_epoch=10,
    epochs=100,
    verbose=1,
    callbacks=callbacks
)

# Save model
model.save(f'{plant}_vgg16.h5')


# Plot
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('loss.png')
plt.show()
