from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter
import numpy as np
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
import os
import pickle
from PIL import Image

# Initialize Tkinter window
main = tkinter.Tk()
main.title("Traffic Sign Classification System")
main.geometry("1300x1200")

# Widgets
text = Text(main, height=15, width=100)
text.pack()
pathlabel = Label(main)
pathlabel.pack()

# Global variables
dataset_path = None
class_labels = None
X, Y = None, None
trainImages, testImages, trainLabels, testLabels = None, None, None, None
model = None

def loadDataset():
    """Load dataset from folder structure"""
    global dataset_path, X, Y, class_labels
    
    text.delete('1.0', END)
    dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
    if not dataset_path:
        messagebox.showwarning("Warning", "No directory selected!")
        return
    
    pathlabel.config(text=dataset_path)
    text.insert(END, f'Dataset path: {dataset_path}\n')
    
    # Get all PNG images
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    if not image_paths:
        messagebox.showerror("Error", "No PNG images found!")
        return
    
    # Load images using PIL (more reliable for PNGs)
    X = []
    Y = []
    class_labels = []
    
    for img_path in image_paths:
        try:
            # Load with PIL and convert to numpy array
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure 3 channels
            img = img.resize((64, 64))
            img_array = np.array(img)
            
            # Get class name from parent folder
            class_name = os.path.basename(os.path.dirname(img_path))
            if class_name not in class_labels:
                class_labels.append(class_name)
            
            X.append(img_array)
            Y.append(class_labels.index(class_name))
            
            text.insert(END, f"Loaded: {os.path.basename(img_path)} (Class: {class_name})\n")
            text.see(END)
            text.update()
            
        except Exception as e:
            text.insert(END, f"Error loading {img_path}: {str(e)}\n")
    
    if not X:
        messagebox.showerror("Error", "No images could be loaded!")
        return
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    class_labels = np.array(class_labels)
    
    # Save processed data
    if not os.path.exists('model'):
        os.makedirs('model')
    np.save('model/X.npy', X)
    np.save('model/Y.npy', Y)
    np.save('model/class_labels.npy', class_labels)
    
    text.insert(END, f"\nSuccessfully loaded {len(X)} images\n")
    text.insert(END, f"Classes found: {', '.join(class_labels)}\n")
    
    # Show sample image
    plt.figure(figsize=(5,5))
    plt.imshow(X[0])
    plt.title(f"Sample Image (Class: {class_labels[Y[0]]})")
    plt.show()

def preprocessDataset():
    """Preprocess and split dataset"""
    global X, Y, trainImages, testImages, trainLabels, testLabels
    
    if X is None:
        messagebox.showerror("Error", "Please load dataset first!")
        return
    
    text.delete('1.0', END)
    text.insert(END, "Preprocessing dataset...\n")
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    Y = to_categorical(Y, num_classes=len(class_labels))
    
    # Split dataset
    trainImages, testImages, trainLabels, testLabels = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    
    text.insert(END, "Dataset preprocessed and split!\n")
    text.insert(END, f"Training images: {trainImages.shape[0]}\n")
    text.insert(END, f"Testing images: {testImages.shape[0]}\n")

def buildModel():
    """Build and train CNN model"""
    global model, trainImages, trainLabels, testImages, testLabels
    
    if trainImages is None:
        messagebox.showerror("Error", "Please preprocess dataset first!")
        return
    
    text.delete('1.0', END)
    text.insert(END, "Building CNN model...\n")
    
    # CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(class_labels), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    # Train model
    if not os.path.exists('model/traffic_model.h5'):
        text.insert(END, "Training model...\n")
        
        checkpoint = ModelCheckpoint('model/traffic_model.h5', 
                                  monitor='val_accuracy',
                                  save_best_only=True,
                                  verbose=1)
        
        history = model.fit(trainImages, trainLabels,
                          epochs=20,
                          batch_size=32,
                          validation_data=(testImages, testLabels),
                          callbacks=[checkpoint],
                          verbose=0)
        
        # Save training history
        with open('model/training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        text.insert(END, "Training completed!\n")
    else:
        model = load_model('model/traffic_model.h5')
        text.insert(END, "Loaded pre-trained model\n")
    
    # Evaluate model
    loss, accuracy = model.evaluate(testImages, testLabels, verbose=0)
    text.insert(END, f"Test Accuracy: {accuracy*100:.2f}%\n")

def classifyImage():
    """Classify a new traffic sign image"""
    global model, class_labels
    
    if model is None:
        messagebox.showerror("Error", "Please train the model first!")
        return
    
    filepath = filedialog.askopenfilename(
        title="Select Traffic Sign Image",
        filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*"))
    )
    
    if not filepath:
        return
    
    try:
        # Load and preprocess image
        img = Image.open(filepath)
        img = img.convert('RGB').resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        class_name = class_labels[class_idx]
        
        # Display results
        text.delete('1.0', END)
        text.insert(END, f"Image: {os.path.basename(filepath)}\n")
        text.insert(END, f"Predicted Class: {class_name}\n")
        text.insert(END, f"Confidence: {confidence*100:.2f}%\n")
        
        # Show the image with prediction
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.title(f"Prediction: {class_name} ({confidence*100:.1f}%)")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        messagebox.showerror("Error", f"Could not process image: {str(e)}")

# Buttons
Button(main, text="Load Dataset", command=loadDataset).pack()
Button(main, text="Preprocess Dataset", command=preprocessDataset).pack()
Button(main, text="Build & Train Model", command=buildModel).pack()
Button(main, text="Classify Image", command=classifyImage).pack()

main.mainloop()