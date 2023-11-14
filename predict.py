import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# Change these paths according to your model and validation directory
model_path = Path('All_vgg16.h5')
validation_dir = Path('validation')

# Load the model
model = load_model(model_path)

# Get class labels
class_labels = sorted(os.listdir(validation_dir))

# Loop through validation images and make predictions
y_true = []
y_pred = []
for label in class_labels:
    class_folder = validation_dir / label
    for img_path in class_folder.glob('*.JPG'):
        true_class = label

        # Load, preprocess and reshape the image
        img = image_utils.load_img(img_path, target_size=(200, 200))
        img = image_utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Make prediction
        outputs = model.predict(img)
        outputs = outputs.reshape(-1)
        pred_idx = np.argmax(outputs)
        pred_class = class_labels[pred_idx]

        # Print prediction for each image
        print(f"Image: {img_path}")
        print(f"True class: {true_class}")
        print(f"Predicted class: {pred_class}\n")

        y_true.append(true_class)
        y_pred.append(pred_class)

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()