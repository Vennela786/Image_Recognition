import cv2 as cv
import numpy as np
import pathlib 
import train_model
import configparser
import predict
import os

config = configparser.RawConfigParser()
config.read('properties.properties')

training_dataset_path = pathlib.Path(config.get('FILES_PATH', 'training_dataset_path'))
trained_recogniser = config.get('FILES_PATH', 'trained_recognizer_export')
img = config.get('FILES_PATH', 'test_image')
recognizer = cv.face.LBPHFaceRecognizer_create()

# To load trained recognizer, if recognizer is not found it will train on the dataset images
if os.path.exists(trained_recogniser):
    recognizer.read(trained_recogniser)
    print("Recognizer Loaded...")
else:
    print("Trained model not found, training on dataset images...")
    recognizer = train_model.train_dataset_images(training_dataset_path, trained_recogniser)
    print("Model trained successfully...")
predict.predict_person(recognizer, img)





