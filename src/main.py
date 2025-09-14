import tkinter as tk
from tkinter import filedialog, Label, Button, BOTTOM
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import os
from pathlib import Path

# Import utility functions
from utils import preprocess_image, predict_traffic_sign

# Define file paths
model_path = Path("models/traffic_sign_model.h5")
labels_path = "data/labels.csv"

# Error handling for loading model
try:
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please verify the path and filename.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Error handling for loading labels
try:
    classes = pd.read_csv(labels_path)
except FileNotFoundError:
    print(f"Labels file not found at {labels_path}. Please verify the path and filename.")
    exit(1)
except Exception as e:
    print(f"Error loading labels: {e}")
    exit(1)

# Initializing the Tkinter GUI
top = tk.Tk()
top.geometry('800x600')
top.title('InsightSign - Traffic Sign Recognition')
top.configure(background='#99cfe0')

label = Label(top, background='#99cfe0', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Function to classify the uploaded image using the utils helpers
def classify(file_path):
    sign = predict_traffic_sign(model, classes, file_path)
    print(sign)
    label.configure(foreground='#0a0a0a', text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#e0aa99', foreground='#0a0a0a', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.6, rely=0.856)

# Function to allow the user to upload an image to classify it
def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im
    label.configure(text='')
    show_classify_button(file_path)

# Main Function GUI setup
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#e0aa99', foreground='#0a0a0a', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="InsightSign - Traffic Sign Detection", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#99cfe0', foreground='#0a0a0a')
heading.pack()
