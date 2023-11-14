import os
import shutil
from pathlib import Path

# Répertoires d'entrée et de sortie
input_dir = Path("images")
output_train_dir = Path("training")
output_valid_dir = Path("validation")

# Crée les répertoires de sortie s'ils n'existent pas
output_train_dir.mkdir(parents=True, exist_ok=True)
output_valid_dir.mkdir(parents=True, exist_ok=True)

# Parcourir chaque sous-répertoire du répertoire "images"
for subdir in input_dir.iterdir():
    if subdir.is_dir():
        
        # Crée les sous répertoires de train et validation si nécessaire
        train_subdir = output_train_dir / subdir.name
        valid_subdir = output_valid_dir / subdir.name
        train_subdir.mkdir(parents=True, exist_ok=True)
        valid_subdir.mkdir(parents=True, exist_ok=True)

        # Les 25 premières images pour la validation, les autres pour l'entraînement
        images = list(subdir.glob("*.*"))
        validation_images = images[:25]
        training_images = images[25:]

        # Copie les images dans le bon répertoire
        for img in training_images:
            shutil.copy(str(img), str(train_subdir / img.name))
        
        for img in validation_images:
            shutil.copy(str(img), str(valid_subdir / img.name))

print("Images réparties entre les dossiers 'training' et 'validation'.")