__package__
import cv2 as cv
import numpy as np
import pathlib as Path
import pickle 


# Load the face cascade classifier for facial recognization
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
#Images dataset path

# To store the training data and labels
training_data = []

#Dictiornay of elemens which has person:unique integer as  key:value pairs 
dict_labels = {}
#**** To pass to train method
labels = []
#This is varibale is used as an counter to increment and use this value to unique integer for dict_labels variable when ever we are training a new person. 
current_label_count = 0

def detect_faces(img):
    """Explain briefly what this method does"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)
    return faces, gray


def train_dataset_images(dataset_path, model_export_path):
    # To store the training data and labels
    training_data = []

    #Dictiornay of elemens which has person:unique integer as  key:value pairs 
    dict_labels = {}
    #**** To pass to train method
    labels = []
    #This is varibale is used as an counter to increment and use this value to unique integer for dict_labels variable when ever we are training a new person. 
    current_label_count = 0
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    for file_path in dataset_path.glob("**/*.jpg"):
        # Load the image and detect faces
        img = cv.imread(str(file_path))
        faces, gray = detect_faces(img)

        # Add each face to the training data array with its corresponding label
        for (x, y, w, h) in faces:
            training_data.append(gray[y:y+h, x:x+w])
            label = str(file_path.parent.name)
            if label not in dict_labels:
                dict_labels[label] = current_label_count
                current_label_count += 1
            labels.append(dict_labels[label])
    face_recognizer.train(training_data, np.array(labels))
    face_recognizer.save(model_export_path)
    with open("label_list.pkl", 'wb') as ll:
        pickle.dump(labels, ll)
    with open("label_dictionary.pkl", 'wb') as sl:
        pickle.dump(dict_labels, sl)

    print("Face Recognizer Model Exported")
    return face_recognizer
