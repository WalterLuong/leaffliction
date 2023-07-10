from sklearn.metrics import classification_report
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def predict_image(image_path):
    model = torch.load(f'data/models/{plant}/export.pkl')
    for image in tqdm(path.glob('**/*.JPG'), total=len(list(path.glob('**/*.JPG')))):
        pred_class, pred_idx, outputs = model.predict(image)
        y_true.append(image.parent.name)
        y_pred.append(pred_class)


if __name__ == '__main__':
    predict_image(path)
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred)
