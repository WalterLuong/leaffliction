from sklearn.metrics import classification_report
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.utils as image
import numpy as np

plant = 'Grape'
path = Path('data/valid', plant)

y_true = []
y_pred = []


def print_prediction(pred_class, pred_idx, outputs, truth):
    print('Prediction: ', pred_class)
    print('Probability: ', outputs[pred_idx].item())
    print('Truth: ', truth)
    print('---------------------------')


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,
                    s=cm[i, j],
                    va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


# def predict_image(image_path):
#     model = torch.load(f'data/models/{plant}/vgg16_1.h5')
#     for image in tqdm(path.glob('**/*.JPG'), total=len(list(path.glob('**/*.JPG')))):
#         pred_class, pred_idx, outputs = model.predict(image)
#         y_true.append(image.parent.name)
#         y_pred.append(pred_class)


def predict_image(image_path):
    model = load_model(f'data/models/{plant}/vgg16_1.h5')
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    img_width, img_height = 200, 200
    # predicting images
    img = image.load_img('data/valid/Grape/Grape_Esca/image (1).JPG', target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes)

    # predicting multiple images at once
    img = image.load_img('data/valid/Grape/Grape_healthy/image (2).JPG', target_size=(img_width, img_height))
    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)

    # pass the list of multiple images np.vstack()
    images = np.vstack([x, y])
    classes = model.predict(images, batch_size=10)

    # print the classes, the images belong to
    print(classes)
    print(classes[0])
    print(classes[0][0])


if __name__ == '__main__':
    predict_image(path)
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred)
