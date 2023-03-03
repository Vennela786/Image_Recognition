import cv2 as cv
import numpy as np
import pathlib 
import train_model
import configparser
import predict

training_dataset_path = pathlib.Path("/Users/moulis_mac/Desktop/AI/dataset")
model_exportpath = "face_recognizer.cv2"
img = '/Users/moulis_mac/Downloads/kane_(103).jpg'
recognizer = cv.face.LBPHFaceRecognizer_create()
# To load trained recognizer, if recognizer is not found it will train on the dataset images
try:
    recognizer.read(model_exportpath)
    print("Recognizer Loaded")
except Exception as e:
    print("Trained model not found, training on dataset images...")
    recognizer = train_model.train_dataset_images(training_dataset_path, model_exportpath)
    print("Model trained successfully")

predict.predict_person(recognizer, img)





