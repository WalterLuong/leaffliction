import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

data_augmentation = tf.keras.Sequential([
    #layers.RandomFlip("horizontal_and_vertical"),
    #layers.RandomRotation(0.2),
    layers.RandomContrast(0.2),
])

result = data_augmentation(image)
_ = plt.imshow(result)
plt.show()
