#%% [markdown]
# Using OpenCV Haar Cascade to detect faces
# 
# # Notes
# 
# Dataset: https://susanqq.github.io/UTKFace/
# 
# The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
# - [age] is an integer from 0 to 116, indicating the age
# - [gender] is either 0 (male) or 1 (female)
# - [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
# - [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

#%%
import cv2
import matplotlib.pyplot as plt
import time
import os
get_ipython().run_line_magic('matplotlib', 'inline')

# Change working directory from the workspace root to the ipynb file location.
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass


#%%
# Base variables
training_file_path = '../resources/haarcascade_frontalface_default.xml'
faces_dataset_path = '../local_resources/UTKFace_cropped_aligned/'
complete_faces_dataset_path = '../local_resources/UTKFace_complete/'
haar_face_cascade = cv2.CascadeClassifier(training_file_path)
example_img_path = '../resources/test1.jpg'
example_img = cv2.imread(example_img_path)


#%% 
# Base functions

def detect_faces(f_cascade, img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = img.copy()
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    return faces

# go over list of faces and draw them as rectangles on original colored img
def print_faces(img, faces):
    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        plt.figure()
    plt.imshow(img_copy)
    return

def load_element_image(element):
    element["image"] = cv2.imread(element["path"])
    return element

def detect_faces_from_element(element):
    element["faces"] = detect_faces(haar_face_cascade, element["image"])
    element["face_count"] = len(element["faces"])
    return element

def filter_age(img_list, range_start, range_end):
    img_list[:] = [x for x in img_list if (range_start <= x["age"] <= range_end)]

#%%
# Run single image test
print_faces(example_img, detect_faces(haar_face_cascade, example_img))


#%%
# Create a list of images
img_name_list = os.listdir(complete_faces_dataset_path)
img_name_list_small = img_name_list[:10]
img_list = []
start_time = time.time()
for img_name in img_name_list_small:
    img_name_split = img_name.split("_")
    img_list.append({
        "name": img_name,
        "path": complete_faces_dataset_path + img_name,
        "age": int(img_name_split[0]),
        "gender": int(img_name_split[1]),
        "race": int(img_name_split[2])
    })
print(time.time() - start_time)

#%%
# Filter list
filter_age(img_list, 90, 100)

#%% 
# Load all images from the list

start_time = time.time()
for img in img_list:
    load_element_image(img)
print(time.time() - start_time)

#%%
# Detect faces from the list
   
start_time = time.time()
for img in img_list:
    detect_faces_from_element(img)
print(time.time() - start_time)


#%%
for img in img_list:
    if img["face_count"] != 1:
        print_faces(img["image"], img["faces"])
        print(img["face_count"])

#%%
