import cv2
import numpy as np

# Chargement de l'image
img = cv2.imread('image.jpg')

# Correction de la couleur
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
img = cv2.merge((cl,a,b))
img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

# Suppression de bruit
img = cv2.GaussianBlur(img, (5, 5), 0)

# Affichage des images
cv2.imshow('Original Image', cv2.imread('image.jpg'))
cv2.imshow('Improved Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
