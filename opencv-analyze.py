import cv2
import numpy as np
from keras.models import load_model
from time import time

# Load the model
model = load_model('keras_model.h5')

# CAMERA can be 0 or 1 based on default camera of your computer.
camera = cv2.VideoCapture(0)

# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()

while True:
    # Grab the webcameras image.
    ret, image = camera.read()
    # Resize the raw image into (224-height,224-width) pixels.
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Make the image a numpy array and reshape it to the models input shape.
    image_tensor = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    #We separate the image to display and the one to process
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    #here I return the image in reverse because by default it is so in order to align it
    image = cv2.flip(image,1)
    # Normalize the image array
    image_tensor = (image_tensor / 127.5) - 1
    # Have the model predict what the current image is. Model.predict
    # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
    # it is the first label and 80% sure its the second label.
    probabilities = model.predict(image_tensor)
    #print(labels[np.argmax(probabilities)])
    #In order to process the information, we transform it
    labs = labels[np.argmax(probabilities)]
    labs = str(labs[0])
    # Print what the highest value probabilitie label / on the opencv image if yoshi is detected or not
    if labs == "1":
        cv2.putText(image, "Yoshi not here", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif labs == "0":
        cv2.putText(image, "Yoshi detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    
    #print(labels[np.argmax(probabilities)])
    # Show the image in a window
    cv2.imshow('Webcam Yoshi detector', image)
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
