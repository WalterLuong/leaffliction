import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import matplotlib.pyplot as plt

# Plant selection and dataset paths
plant = 'Apple'
dataset_train_path = Path('data', 'images', plant)
dataset_valid_path = Path('data', 'valid', plant)


# Load dataset
traindata = ImageDataGenerator().flow_from_directory(
    directory=dataset_train_path, target_size=(200, 200))
testdata = ImageDataGenerator().flow_from_directory(
    directory=dataset_valid_path, target_size=(200, 200))


# Model definition (pretrained VGG16)
vgg = VGG16(input_shape=[200, 200] + [3],
            weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
outputs = Dense(8, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=outputs)

opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt,
              loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

# Train
checkpoint = ModelCheckpoint(f'{plant}_pretrained_vgg16.h5', monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max', save_freq=1)
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=20, verbose=1, mode='max')
callbacks = [checkpoint, early_stopping]

hist = model.fit(traindata, validation_data=testdata, batch_size=2, steps_per_epoch=10, epochs=100,
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
