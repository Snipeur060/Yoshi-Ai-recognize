import os
import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('../keras_model/keras_model.h5')

# The folder containing the images
input_folder = 'img/'
# The folder where the images will be moved if their probability is greater than 0.5
output_folder = 'outpoutanalyze/'

# Iterate over all the files in the input folder
for filename in os.listdir(input_folder):
    # Get the full path of the file
    filepath = os.path.join(input_folder, filename)
    # Load the image
    print(filename)
    image = cv2.imread(filepath)
    # Resize the image to the input shape of the model
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    image = (image / 127.5) - 1
    # Have the model predict what the current image is. Model.predict
    # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
    # it is the first label and 80% sure its the second label.
    probabilities = model.predict(image)
    # Get the probability of the image belonging to the first class
    probability = probabilities[0][0]
    # If the probability is greater than 0.5
    if probability > 0.7:
        # Move the file to the output folder
        os.rename(filepath, os.path.join(output_folder, filename))
