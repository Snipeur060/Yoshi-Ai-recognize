__version__ = '0.1.1'

import tkinter
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog

import cv2
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps  # Install pillow instead of PIL
from pathlib import Path

# Charger le modèle keras
model = load_model("../keras_model/keras_model.h5")
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

print("""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⠾⠟⠻⠶⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣤⣄⣀⣾⠁⣀⡀⠀⠀⠈⠹⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⠋⣁⠈⠉⢻⣷⠿⠛⠛⠿⣦⣄⠀⠈⢿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⠛⠛⠻⣷⣾⡏⠀⠀⠀⠀⠈⠻⣷⡀⠈⢿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠈⢿⡇⠀⠀⠀⠀⠀⢀⡘⢿⡄⠈⠻⣦⣀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠠⡶⣾⣧⠀⠀⠀⠀⢰⡏⢹⣮⣿⡀⠀⠈⠻⡍⠉⠉⠙⢿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⣀⠀⠀⣷⣿⣿⠀⠀⠀⠀⠈⢿⣿⣿⢹⣧⠀⠀⠀⠻⣄⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⡶⠟⠛⠛⠛⠛⠛⠛⠛⠻⢿⣧⣀⠀⠀⠀⠈⠛⣟⣼⠿⠒⠂⠀⠀⠉⠙⠓⢦⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣤⡾⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠳⢦⡄⣠⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⣴⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣆⠈⠛⢿⡀⠀⠀⠀⠀⠀⠀⢠⡾⠛⢿⣷⠀⠀⠀⠀⠀⠀⠀
⠀⢠⣾⠋⠀⠀⠀⠀⢰⡟⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡀⠀⣾⣁⣀⣀⠀⠀⠀⠀⣿⡇⠀⢸⣿⠀⣤⣴⣤⡀⠀⠀
⢠⣾⠃⠀⠉⠂⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣇⣾⢣⣿⠛⠛⠿⣦⡀⢀⣿⠁⠀⣾⢣⡿⠋⠀⢹⣿⠀⠀
⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠃⠀⢿⣧⡀⠀⢹⣷⡟⠉⠀⠚⢿⣿⠁⠀⣠⡾⠃⠀⠀
⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⠀⠀⠀⠙⣿⡄⠀⠉⠻⣆⠀⠀⠀⠙⢶⣿⣫⣤⣤⣤⣄
⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠛⣧⠀⠀⠀⢻⡇⠀⠀⠀⠈⠀⠀⠀⠀⠀⠛⠋⠀⠀⠈⣻
⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡟⠀⠈⣷⠀⠀⢸⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣶⣶⠾⠟
⠈⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡾⢁⣀⡴⠋⠀⣀⣾⠟⠁⠀⠀⠀⠀⠀⠀⢀⣴⡿⠋⠀⠀⠀⠀
⠀⠸⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠏⠈⠙⠳⣶⠤⠀⠀⠀⠀⢀⡾⠟⠉⠁⠀⣠⠶⠋⠀⠀⠀⣠⣶⣤⣤⣴⣶⠿⠋⠀⠀⠀⠀⠀⠀
⠀⠀⠹⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠃⠀⠀⠀⢀⡟⠀⠀⠀⠀⢠⣾⣅⠀⢀⣴⠾⠋⠀⠀⠀⢀⣴⠟⠁⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⡇⠀⠀⠀⠀⣸⠃⠀⠀⠀⠀⠋⠈⢻⣷⠟⠉⠀⠀⠀⠀⣠⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠛⢷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⠾⣯⠀⠹⣿⠀⠀⠀⣠⠟⠀⠀⠀⠀⠀⠀⢀⣼⠃⠀⠀⠀⠀⢠⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠉⠻⠷⣶⣦⣄⣀⣀⣀⣠⣤⡶⠶⠛⠉⠀⠀⠘⢧⡀⠙⠷⠤⠴⠋⠀⠀⢀⡴⠇⠀⠀⠈⠁⠀⠀⠀⠀⣠⡿⣧⣄⡀⠀⢀⣴⣶⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣬⣭⣭⣉⣉⡉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱⣦⣀⠀⠀⣀⣠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⢿⡀⠈⢿⢻⣾⠟⠉⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⠁⠀⠈⠉⠉⠛⠛⠛⠛⠛⠻⠿⠶⠶⠚⠉⠉⠉⣿⠟⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣦⠻⣦⣸⡆⣿⠀⠀⠀⠈⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣴⡶⠶⠶⠶⠿⢷⣦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣌⣉⣢⡿⠀⠀⠀⠀⢻⡇⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣿⡀⠀⠀⠀⠀⠀⠀⠉⠙⡷⢤⣄⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣼⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠀⠀⠀⠀⠀⠸⣿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠻⣷⣤⣤⣤⣤⠤⠴⠒⠛⢧⡀⠀⠀⠀⠀⠀⠀⠀⣰⡿⠋⠉⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣴⠟⠁⠀⠀⠀⠀⢀⣠⡼⠃⠀⠀⠀⠀⠀⢀⣴⠏⠀⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣿⣤⣀⢀⣠⣴⡾⠋⠁⠀⠀⣠⡶⠶⠶⠿⠛⠁⠀⠀⠀⠀⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡗⠀⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠈⠉⠛⠋⢹⡏⠀⠀⠀⣠⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡟⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠿⣦⣤⠾⠋⠀⠀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀⠀⠀⠀⠀⠀⠀⠀⢰⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⡿⠛⠛⠛⠻⢿⣶⣄⠀⠀⢀⣴⡿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠏⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⣾⣏⠀⠈⢿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡏⢿⠟⢶⣤⣄⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⣛⣛⣛⡛⠿⣦⣄⠀⠀⠈⢿⡝⠛⠻⢶⣤⡙⢿⣷⣤⣀⠀⠀⠀⠀⠀⠀⠀⢙⣧⡀⠀⠀⠀⠀⠀⠀⠀⣸⠇⢸⠃⠀⠻⣌⢳⡄⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⡟⠉⠉⠙⠛⠷⣤⣙⠿⣦⠀⠈⠛⠓⠆⠀⠙⢿⣦⣼⡿⠿⠿⠿⣿⣿⡿⠿⠿⠛⠛⠻⣤⡀⠀⠀⠀⢀⡴⠃⠀⠙⠀⠀⠀⠙⣌⢷⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⡀⠀⠀⠀⠀⠀⠛⢷⣌⢷⣄⠀⠀⠀⠀⢤⠀⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣷⣦⣞⣋⣀⣠⢞⠀⠀⠀⠀⠀⠸⣎⢳⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢿⣦⡀⠀⠀⠀⠀⠀⠹⣷⡹⣦⡀⠀⠀⠈⣷⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⢹⣏⣠⡾⠀⠀⠀⠀⠀⠀⢸⡌⢳
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⣶⣤⣀⣀⡀⢀⣈⡻⣮⡻⣷⣀⠀⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣻⠏⠀⠀⠀⠀⠀⠀⠀⠈⣷⠸
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⢿⣟⠉⠀⠈⢻⣎⣿⣄⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣧⣄⡀⢀⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣆⠀⠀⠀⠀⠀⠀⢀⣼⠇⡸
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡙⠛⠛⣛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢷⣤⣄⣠⣤⣶⣛⣥⡾⠃
""")

def select_image():
    # Load the labels
    class_names = open('../keras_model/labels.txt', 'r').readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    file_path_to_analyze = filedialog.askopenfilename()
    # Replace this with the path to your image
    image = Image.open(file_path_to_analyze).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    #image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #probability = round(prediction[0][0],2)
    #echec = round(1-prediction[0][0],2)
    index = np.argmax(prediction)

    class_name = class_names[index]
    if class_name[0] == "0":
        result_name = "Yoshi"
    else:
        result_name = "Rien"
    canvas.itemconfig(nameclass, text=result_name)
    canvas.itemconfig(numaccur, text= round(prediction[0][0],2))
    canvas.itemconfig(numfail, text=round(1-prediction[0][0],2))
    canvas.itemconfig(numlabel, text=class_name[0])
    canvas.itemconfig(path, text=file_path_to_analyze)

    #resultname.configure(text=class_name)


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame0/")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()
window.title("Yoshi Ai Recognition")
window.geometry("1205x862")
window.configure(bg="#1A1C48")


canvas = Canvas(
    window,
    bg="#1A1C48",
    height=862,
    width=1205,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    784.0,
    431.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    896.65771484375,
    241.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    835.4384765625,
    274.2682800292969,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    897.0,
    246.0,
    image=image_image_4
)

canvas.create_text(
    876.65771484375,
    211.03421020507812,
    anchor="nw",
    text="Résultat",
    fill="#FFFFFF",
    font=("Lato Regular", 12 * -1)
)
#resultname = tkinter.Label(window, text="Aucune image sélectionnée", anchor="nw",
#                           bg="#ab23ff",
#                           font=("Lato Medium", 32 * -1))
#resultname.place(x=819.65771484375, y=231.03421020507812)

nameclass = canvas.create_text(
    859.65771484375,
    231.03421020507812,
    text="NaN",
    anchor="nw",
    fill="#FFFFFF",
    font=("Lato Medium", 32 * -1)
)

canvas.create_text(
    870.65771484375,
    275.0342102050781,
    anchor="nw",
    text="Result",
    fill="#FFFFFF",
    font=("Lato Medium", 20 * -1)
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    705.0,
    246.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    704.4384765625,
    241.0,
    image=image_image_6
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    644.21923828125,
    283.0342102050781,
    image=image_image_7
)

canvas.create_text(
    666.4384765625,
    211.03421020507812,
    anchor="nw",
    text="Text (labels.txt)",
    fill="#FFFFFF",
    font=("Lato Regular", 12 * -1)
)

numlabel = canvas.create_text(
    680.4384765625,
    231.03421020507812,
    anchor="nw",
    text="NaN",
    fill="#FFFFFF",
    font=("Lato Medium", 32 * -1)
)

canvas.create_text(
    677.4384765625,
    275.0342102050781,
    anchor="nw",
    text="Labels",
    fill="#FFFFFF",
    font=("Lato Medium", 20 * -1)
)



canvas.create_text(
    875.4384765625,
    626.0342102050781,
    anchor="nw",
    text="Chemin du fichier:",
    fill="#FFFFFF",
    font=("Lato Medium", 20 * -1)
)

path = canvas.create_text(
    832.4384765625,
    654.0342102050781,
    anchor="nw",
    text="Ancun fichier selectionné",
    fill="#FFFFFF",
    font=("Lato Medium", 20 * -1)
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    513.0,
    246.0,
    image=image_image_8
)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    458.841796875,
    298.1943359375,
    image=image_image_9
)

canvas.create_text(
    480.21923828125,
    211.03421020507812,
    anchor="nw",
    text="Taux d’Echec",
    fill="#FFFFFF",
    font=("Lato Regular", 12 * -1)
)

numfail = canvas.create_text(
    485.21923828125,
    231.03421020507812,
    anchor="nw",
    text="NaN",
    fill="#FFFFFF",
    font=("Lato Medium", 32 * -1)
)

canvas.create_text(
    495.21923828125,
    275.0342102050781,
    anchor="nw",
    text="Loss",
    fill="#FFFFFF",
    font=("Lato Medium", 20 * -1)
)

image_image_10 = PhotoImage(
    file=relative_to_assets("image_10.png"))
image_10 = canvas.create_image(
    513.21923828125,
    241.0,
    image=image_image_10
)

image_image_11 = PhotoImage(
    file=relative_to_assets("image_11.png"))
image_11 = canvas.create_image(
    321.0,
    246.0,
    image=image_image_11
)

image_image_12 = PhotoImage(
    file=relative_to_assets("image_12.png"))
image_12 = canvas.create_image(
    320.21923828125,
    241.0,
    image=image_image_12
)

image_image_13 = PhotoImage(
    file=relative_to_assets("image_13.png"))
image_13 = canvas.create_image(
    260.0,
    283.0342102050781,
    image=image_image_13
)

image_image_14 = PhotoImage(
    file=relative_to_assets("image_14.png"))
image_14 = canvas.create_image(
    257.841796875,
    288.0094566345215,
    image=image_image_14
)

image_image_15 = PhotoImage(
    file=relative_to_assets("image_15.png"))
image_15 = canvas.create_image(
    250.21923828125,
    257.0342102050781,
    image=image_image_15
)

canvas.create_text(
    279.21923828125,
    211.03421020507812,
    anchor="nw",
    text="Taux de réussite",
    fill="#FFFFFF",
    font=("Lato Regular", 12 * -1)
)

numaccur = canvas.create_text(
    290.21923828125,
    231.03421020507812,
    anchor="nw",
    text="NaN",
    fill="#FFFFFF",
    font=("Lato Medium", 32 * -1)
)

canvas.create_text(
    281.21923828125,
    275.0342102050781,
    anchor="nw",
    text="Accuracy",
    fill="#FFFFFF",
    font=("Lato Medium", 20 * -1)
)

image_image_16 = PhotoImage(
    file=relative_to_assets("image_16.png"))
image_16 = canvas.create_image(
    266.0,
    439.0,
    image=image_image_16
)

image_image_17 = PhotoImage(
    file=relative_to_assets("image_17.png"))
image_17 = canvas.create_image(
    265.5,
    433.09136962890625,
    image=image_image_17
)

image_image_18 = PhotoImage(
    file=relative_to_assets("image_18.png"))
image_18 = canvas.create_image(
    250.0,
    432.00006103515625,
    image=image_image_18
)

canvas.create_text(
    246.0,
    407.12554931640625,
    anchor="nw",
    text="Version",
    fill="#FFFFFF",
    font=("Lato Regular", 12 * -1)
)

canvas.create_text(
    240.0,
    427.12554931640625,
    anchor="nw",
    text=__version__,
    fill="#FFFFFF",
    font=("Lato Bold", 24 * -1)
)

canvas.create_text(
    239.0,
    467.12554931640625,
    anchor="nw",
    text="Version",
    fill="#FFFFFF",
    font=("Lato Bold", 16 * -1)
)

image_image_19 = PhotoImage(
    file=relative_to_assets("image_19.png"))
image_19 = canvas.create_image(
    424.0,
    439.0,
    image=image_image_19
)

image_image_20 = PhotoImage(
    file=relative_to_assets("image_20.png"))
image_20 = canvas.create_image(
    781.0,
    439.0,
    image=image_image_20
)

image_image_21 = PhotoImage(
    file=relative_to_assets("image_21.png"))
image_21 = canvas.create_image(
    939.0,
    439.0,
    image=image_image_21
)

canvas.create_text(
    197.0,
    553.0,
    anchor="nw",
    text="CONNECTIONS",
    fill="#9499C3",
    font=("Lato SemiBold", 14 * -1)
)

canvas.create_text(
    197.0,
    592.0,
    anchor="nw",
    text="Manual Rigs",
    fill="#9499C3",
    font=("Lato Regular", 12 * -1)
)

canvas.create_text(
    301.0,
    592.0,
    anchor="nw",
    text="Connected",
    fill="#FFFFFF",
    font=("Lato Bold", 12 * -1)
)

canvas.create_text(
    197.0,
    620.0,
    anchor="nw",
    text="Changelog",
    fill="#9499C3",
    font=("Lato Regular", 12 * -1)
)

canvas.create_text(
    301.0,
    620.0,
    anchor="nw",
    text="Connected",
    fill="#FFFFFF",
    font=("Lato Bold", 12 * -1)
)

canvas.create_text(
    197.0,
    648.0,
    anchor="nw",
    text="KerasModel",
    fill="#9499C3",
    font=("Lato Regular", 12 * -1)
)

canvas.create_text(
    301.0,
    648.0,
    anchor="nw",
    text="Connected",
    fill="#FFFFFF",
    font=("Lato Bold", 12 * -1)
)

canvas.create_text(
    197.0,
    676.0,
    anchor="nw",
    text="UI",
    fill="#9499C3",
    font=("Lato Regular", 12 * -1)
)

canvas.create_text(
    301.0,
    676.0,
    anchor="nw",
    text="Connected",
    fill="#FFFFFF",
    font=("Lato Bold", 12 * -1)
)

canvas.create_rectangle(
    197.0,
    583.0,
    365.0,
    584.0,
    fill="#FFFFFF",
    outline="")

image_image_22 = PhotoImage(
    file=relative_to_assets("image_22.png"))
image_22 = canvas.create_image(
    606.0,
    515.0,
    image=image_image_22
)

image_image_23 = PhotoImage(
    file=relative_to_assets("image_23.png"))
image_23 = canvas.create_image(
    153.0,
    440.0,
    image=image_image_23
)

canvas.create_rectangle(
    166.0,
    845.111083984375,
    1734.0,
    862.0,
    fill="#1A1C48",
    outline="")

canvas.create_rectangle(
    214.0,
    767.0,
    1022.0,
    862.0,
    fill="#1A1B48",
    outline="")

image_image_24 = PhotoImage(
    file=relative_to_assets("image_24.png"))
image_24 = canvas.create_image(
    617.0,
    814.111083984375,
    image=image_image_24
)

canvas.create_rectangle(
    584.0,
    854.0,
    652.0,
    858.2222290039062,
    fill="#FFFFFF",
    outline="")

canvas.create_text(
    511.0,
    90.0,
    anchor="nw",
    text="Yoshi Ai Recognize",
    fill="#FFFFFF",
    font=("Lato SemiBold", 26 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: select_image(),
    relief="flat",
    bg="#1A1C48"
)
button_1.place(
    x=546.0,
    y=688.0,
    width=114.0,
    height=56.0
)
window.wm_attributes('-transparentcolor', '#ab23ff')

window.resizable(False, False)
window.mainloop()
