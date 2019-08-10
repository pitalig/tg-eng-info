#%%
import json

with open('src/result_faces_scaleFactor_13_minNeighbors_5_minSize_30_30.json') as json_file:
    result = json.load(json_file)
    print(len(result["raw_result"]))

#%%
# Count faces

result["count_imgs"] = len(result["raw_result"])
result["count_0"] = 0
result["count_1"] = 0
result["count_2"] = 0
for img in result["raw_result"]:
    if img["face_count"] == 0:
      result["count_0"] += 1
    elif img["face_count"] == 1:
      result["count_1"] += 1
    else:
      result["count_2"] += 1

print(str(result["count_0"]) + " images with 0 faces (" + str(result["count_0"] * 100 / result["count_imgs"]) + "%)")
print(str(result["count_1"]) + " images with 1 faces (" + str(result["count_1"]*100/result["count_imgs"]) + "%)")
print(str(result["count_2"]) + " images with 2 or more faces (" + str(result["count_2"]*100/result["count_imgs"]) + "%)")

#%% [markdown]
