#%%
import pandas as pd
import numpy as np
import json

if 1 == 1:
  faces_path = 'src/result_faces_scaleFactor_13_minNeighbors_5_minSize_30_30.json'
  no_faces_path = 'src/result_no_faces_scaleFactor_13_minNeighbors_5_minSize_30_30.json'
  test_check = 1
else:
  faces_path = 'src/result_faces_scaleFactor_105_minNeighbors_3_minSize_30_30.json'
  no_faces_path = 'src/result_no_faces_scaleFactor_105_minNeighbors_3_minSize_30_30.json'
  test_check = 1

with open(faces_path) as json_file:
  result_faces = json.load(json_file)
result_faces["raw_result"] = result_faces["raw_result"][:17130]
#print(len(result_faces["raw_result"]))

with open(no_faces_path) as json_file:
  result_no_faces = json.load(json_file)
result_no_faces["raw_result"] = result_no_faces["raw_result"][:17130]
#print(len(result_no_faces["raw_result"]))

#%%
# Count faces

def add_counts(result_data):
  result_data["count_imgs"] = len(result_data["raw_result"])
  result_data["count_0"] = 0
  result_data["count_1"] = 0
  result_data["count_2"] = 0
  result_data["count_1+"] = 0
  for img in result_data["raw_result"]:
      if img["face_count"] == 0:
        result_data["count_0"] += 1
      elif img["face_count"] == 1:
        result_data["count_1"] += 1
        result_data["count_1+"] += 1
      else:
        result_data["count_2"] += 1
        result_data["count_1+"] += 1
  print(str(result_data["count_0"]) + " images with 0 faces (" + str(result_data["count_0"] * 100 / result_data["count_imgs"]) + "%)")
  print(str(result_data["count_1"]) + " images with 1 faces (" + str(result_data["count_1"]*100/result_data["count_imgs"]) + "%)")
  print(str(result_data["count_1+"]) + " images with 1 or more faces (" + str(result_data["count_1+"]*100/result_data["count_imgs"]) + "%)")
  print(str(result_data["count_2"]) + " images with 2 or more faces (" + str(result_data["count_2"]*100/result_data["count_imgs"]) + "%)")

print("=== Faces Dataset ===")
add_counts(result_faces)

print("=== No Faces Dataset ===")
add_counts(result_no_faces)

#%% Confusion matrix with sklearn
from sklearn.metrics import confusion_matrix

y_true = (["face"] * result_faces["count_imgs"]) + (["nao_face"] * result_no_faces["count_imgs"])
y_pred = (["face"] * result_faces["count_1+"]) + (["nao_face"] * result_faces["count_0"]) + (["face"] * result_no_faces["count_1+"]) + (["nao_face"] * result_no_faces["count_0"])

assert len(y_true) == len(y_pred), "Mesmo número de items existentes e previstos"

matrix = confusion_matrix(y_true, y_pred, labels=["face", "nao_face"])

tp, fn, fp, tn = [i/len(y_true) for i in matrix.ravel()]

result_print = np.array([[tp, fn, (tp+fn)],
                         [fp, tn, (fp+tn)],
                         [(tp+fp), (fn+tn), (tp*fn*fp*tn)]]) *100

print("sensitividade")
print(tp / (tp + fn))
print("especificidade")
print(tn / (tn + fp))
pd.DataFrame(result_print, columns=["(B) Face (previsto)", "(xB) Não Face (previsto)", ""], index=["(A) Face (real)", "(xA) Não Face (real)", ""])

#%% Calculate model profit

h = 2 # Manual analysis cost for each picture
v = 250.37 # Approved customer value

g0 = (tp+fn)*v - (tp+fp+fn+tn)*h
g1 = tp*v - (tp+fp)*h

for i in range(1,100):
  g1 += (0.8**i)*fn*(tp*v - (tp+fp)*h)

print(g1 - g0)

#print((fn+tn)/fn)
#print(c_value/h_cost)

#print(h_cost*(fn+tn) - fn*c_value)

# %%
