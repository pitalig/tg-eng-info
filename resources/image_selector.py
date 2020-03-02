# %%

# Regras:
# Olhos, nariz e boca visíveis
# Images com boa definição e foco
# Face frontal centralizada olhando para a camera
# Apenas uma face na images (imagem com segunda face, mesmo que cortada, é inválida)
# Não pode estar utilizando chapéu ou boné
# Pode estar utiliizando óculos de grau mas não oculos escuros

import cv2
import os
import shutil 
import random
from IPython.display import clear_output
from matplotlib import pyplot as plt
%matplotlib inline

# %%

original_folder = "../local_resources/dataset_UTKFaces_age_18_to_60"
selected_folder = "../local_resources/selected_images"
selected_invalid_folder = "../local_resources/selected_invalid_images"

# %%
# Read images from folder

# images = os.listdir(original_folder)
# random.shuffle(images)

#%%
# Read images from saved list

with open ('img_list', 'rb') as fp:
    images = pickle.load(fp)
    
# %%
# Run selector

i = 905

while i < len(images):

    clear_output()

    img_name = images[i]

    print(i)
    print(len(os.listdir(selected_folder)))
    print(len(os.listdir(selected_invalid_folder)))
    
    img_path = original_folder + "/" + img_name
    img = cv2.imread(img_path)222
    
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    print(img_path)
    
    response = None
    while response not in {"1", "2", "3", "4", "5"}:
        response = input("Please enter 1 (select), 2 (next), 3 (select invalid), 4 (break), 5 (back): ")
    
    if response == "1":
        shutil.copyfile(img_path, selected_folder + "/" + img_name)
    elif response == "3":
        shutil.copyfile(img_path, selected_invalid_folder + "/" + img_name)
    elif response == "4":
        break
    elif response == "5":
        i -= 2

    i += 1
