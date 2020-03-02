# Regras:
# Olhos, nariz e boca visíveis
# Images com boa definição e foco
# Face frontal centralizada olhando para a camera
# Apenas uma face na images (imagem com segunda face, mesmo que cortada, é inválida)

import cv2
import os
import shutil 
import random
from IPython.display import clear_output

original_folder = "../local_resources/dataset_UTKFaces_age_18_to_60"
selected_folder = "../local_resources/selected_images"
selected_invalid_folder = "../local_resources/selected_invalid_images"

folders = os.listdir(original_folder)
random.shuffle(folders)
for folder in folders:
  print(len(os.listdir(selected_folder)))
  
  for img_name in os.listdir(original_folder + folder):
    img_path = original_folder + folder + "/" + img_name
    img = cv2.imread(img_path)
    cv2.imshow(img)
    print(img_path)
    response = None
    while response not in {"1", "2", "3", "4"}:
      response = input("Please enter 1 or 2 or 3 or 4: ")
    clear_output(wait=True)
    if response == "1":
      shutil.copyfile(img_path, selected_folder + "/" + img_name)
      break
    if response == "3":
      shutil.copyfile(img_path, selected_invalid_folder + "/" + img_name)
      break
    if response == "4":
      break