#%%
import pandas as pd
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
print(len(result_faces["raw_result"]))

with open(no_faces_path) as json_file:
  result_no_faces = json.load(json_file)
result_no_faces["raw_result"] = result_no_faces["raw_result"][:17130]
print(len(result_no_faces["raw_result"]))

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

#%% [markdown]

# |                      | (B) Face (previsto) | (xB) Não Face (previsto) |      |
# |----------------------|---------------------|--------------------------|------|
# | (A) Face (real)      | B_A                 | xB_A                     | P_A  |
# | (xA) Não Face (real) | B_xA                | xB_xA                    | P_xA |
# |                      | P_B                 | P_xB                     | 1    |

#%%

results_total = result_faces["count_imgs"] + result_no_faces["count_imgs"] # Universo
P_A = result_faces["count_imgs"] / results_total
P_xA = result_no_faces["count_imgs"] / results_total
P_B = (result_no_faces["count_1+"] + result_faces["count_1+"]) / results_total
P_xB = (result_no_faces["count_0"] + result_faces["count_0"]) / results_total
print([P_A, P_xA, sum([P_A, P_xA])])
assert sum([P_A, P_xA]) == 1, "Todas imagens devem ter ou nao faces"
print([P_B, P_xB, sum([P_B, P_xB])])
assert sum([P_B, P_xB]) == 1, "Todas previsoes devem ter ou nao faces"

#%%

P_A_n_B = result_faces["count_1+"] / results_total
P_xA_n_B = result_no_faces["count_1+"] / results_total
P_A_n_xB = result_faces["count_0"] / results_total
P_xA_n_xB = result_no_faces["count_0"] / results_total
print([P_A_n_B, P_xA_n_B, P_A_n_xB, P_xA_n_xB, sum([P_A_n_B, P_xA_n_B, P_A_n_xB, P_xA_n_xB])])
assert sum([P_A_n_B, P_xA_n_B, P_A_n_xB, P_xA_n_xB]) == test_check, "Todas intersecoes somam 1"

#%%

P_B_given_A = P_A_n_B / P_A
P_xB_given_A = P_A_n_xB / P_A
P_B_given_xA = P_xA_n_B / P_xA
P_xB_given_xA = P_xA_n_xB / P_xA
print([P_B_given_A, P_xB_given_A, sum([P_B_given_A, P_xB_given_A])])
assert sum([P_B_given_A, P_xB_given_A]) == 1, "Uma nao face pode so ser prevista como face ou nao"
print([P_B_given_xA, P_xB_given_xA, sum([P_B_given_xA, P_xB_given_xA])])
assert sum([P_B_given_xA, P_xB_given_xA]) == 1, "Uma face pode so ser prevista como face ou nao"

print("sensitividade")
print(P_B_given_A)
print("especificidade")
print(P_xB_given_xA)

result_print = [[P_A_n_B*100, P_A_n_xB*100, P_A*100],
                [P_xA_n_B*100, P_xA_n_xB*100, P_xA*100],
                [P_B*100, P_xB*100, 100]]
pd.DataFrame(result_print, columns=["(B) Face (previsto)", "(xB) Não Face (previsto)", ""], index=["(A) Face (real)", "(xA) Não Face (real)", ""])

#%%
