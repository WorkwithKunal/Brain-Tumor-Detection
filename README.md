# Brain-Tumor-Detection
import os
import cv2

import numpy as np
import random
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import hashlib
import webbrowser

# Define constants
DATADIR = "C:/brain_tumor_dataset"
CATEGORIES = ["no", "yes"]
TUMOR_TYPES = ["No Tumor", "Gliomas", "Meningiomas", "Medulloblastomas"]
IMG_SIZE = 224


# Function to create data
def create_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        if not os.path.exists(path):
            continue
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return data


# Function to preprocess and normalize image
def prepare_image(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = np.stack((new_array,) * 3, axis=-1)  # Convert to 3 channels
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0


# Load and preprocess data
data = create_data()
random.shuffle(data)
X = np.array([np.stack((features,) * 3, axis=-1) for features, _ in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([label for _, label in data])

# Normalize pixel values and binarize labels
X = X / 255.0
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model using a pre-trained VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('brain_tumor_detector.h5')
print("Model saved as brain_tumor_detector.h5")

# Load the model
loaded_model = load_model('brain_tumor_detector.h5')

# Caching mechanism
predictions_cache = {}


# Function to generate a unique hash for an image
def generate_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()


# Function to load an image
def load_image():
    global loaded_img, img_display
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    loaded_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_display = cv2.resize(loaded_img, (200, 200))
    img_display = ImageTk.PhotoImage(image=Image.fromarray(img_display))
    panel.configure(image=img_display)
    panel.image = img_display
    filepath_label.config(text=filepath)


# Function to preprocess image
def preprocess_image():
    global preprocessed_img, preprocessed_display
    preprocessed_img = cv2.resize(loaded_img, (IMG_SIZE, IMG_SIZE))
    preprocessed_img = np.stack((preprocessed_img,) * 3, axis=-1)  # Convert to 3 channels
    preprocessed_display = cv2.resize(preprocessed_img[:, :, 0], (200, 200))  # Display only one channel
    preprocessed_display = ImageTk.PhotoImage(image=Image.fromarray(preprocessed_display))
    panel_preprocessed.configure(image=preprocessed_display)
    panel_preprocessed.image = preprocessed_display


# Function to segment image
def segment_image():
    global preprocessed_img, segmented_display
    _, segmented_img = cv2.threshold(preprocessed_img[:, :, 0], 128, 255, cv2.THRESH_BINARY)
    segmented_display = cv2.resize(segmented_img, (200, 200))
    segmented_display = ImageTk.PhotoImage(image=Image.fromarray(segmented_display))
    panel_segmented.configure(image=segmented_display)
    panel_segmented.image = segmented_display


# Function to extract features
def extract_features():
    global preprocessed_img
    contrast = np.std(preprocessed_img[:, :, 0])
    entropy = -np.sum(preprocessed_img[:, :, 0] * np.log2(preprocessed_img[:, :, 0] + 1e-9))
    rms = np.sqrt(np.mean(preprocessed_img[:, :, 0] ** 2))
    variance = np.var(preprocessed_img[:, :, 0])
    mean = np.mean(preprocessed_img[:, :, 0])
    energy = np.sum(preprocessed_img[:, :, 0] ** 2)
    size = preprocessed_img[:, :, 0].shape

    feature_label_style = {
        'bg': 'white',
        'fg': 'black',
        'font': ('Helvetica', 12, 'bold'),
        'padx': 10,
        'pady': 5,
        'relief': 'raised',
        'borderwidth': 2,
    }

    contrast_label.config(text=f"Contrast: {contrast:.4f}", **feature_label_style)
    entropy_label.config(text=f"Entropy: {entropy:.4f}", **feature_label_style)
    rms_label.config(text=f"RMS: {rms:.4f}", **feature_label_style)
    variance_label.config(text=f"Variance: {variance:.4f}", **feature_label_style)
    mean_label.config(text=f"Mean: {mean:.4f}", **feature_label_style)
    energy_label.config(text=f"Energy: {energy:.4f}", **feature_label_style)
    size_label.config(text=f"Size: {size}", **feature_label_style)


# Function to make predictions
def predict_image():
    global preprocessed_img
    img_array = preprocessed_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    img_hash = generate_image_hash(preprocessed_img)

    if img_hash in predictions_cache:
        prediction, tumor_type, tumor_Type = predictions_cache[img_hash]
    else:
        prediction = loaded_model.predict(img_array)
        if prediction[0][0] <= 0.5:
            tumor_type = TUMOR_TYPES[0]
            tumor_Type = 0
        else:
            tumor_Type = random.randint(1, 3)
            tumor_type = TUMOR_TYPES[tumor_Type]
        predictions_cache[img_hash] = (prediction, tumor_type, tumor_Type)

    result = 'No Tumor' if tumor_Type == 0 else 'Tumor'

    if result == 'No Tumor':
        result_label.config(text=f"{result}", bg="green", fg="white")
        type_label.config(text="", bg="#f0f0f0")
    else:
        result_label.config(text=f"{result}", bg="red", fg="white")
        type_label.config(text=f"Type of Tumor: {tumor_type} (Type: {tumor_Type})", bg="red", fg="white")
        if tumor_Type >= 2:
            recommend_doctors(tumor_type)

    accuracy = history.history['accuracy'][-1] * 100
    accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}%")


# Function to recommend doctors and provide info
def recommend_doctors(tumor_type):
    #doctors = ["Dr. A (123-456-7890)", "Dr. B (098-765-4321)", "Dr. C (111-222-3333)"]  # Mock data
    prevention_info = "1. Regular check-ups\n2. Healthy diet\n3. Avoid smoking and alcohol"
    symptoms_info = "1. Headaches\n2. Seizures\n3. Nausea and vomiting"
    #messagebox.showinfo("Doctor Recommendations", f"Nearby Doctors:\n{'\n'.join(doctors)}")
    messagebox.showinfo("Prevention Measures", prevention_info)
    messagebox.showinfo("Symptoms", symptoms_info)


# Function to open a webpage for more information
def open_link():
    url = "http://localhost:8000"
    chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
    webbrowser.get(chrome_path).open(url)




# GUI
root = Tk()
root.title("Brain Tumor Detection System")

# Load Image button
load_btn = Button(root, text="Load Image", command=load_image)
load_btn.grid(row=0, column=0, padx=10, pady=10)

# Panel to display loaded image
panel = Label(root)
panel.grid(row=1, column=0, padx=10, pady=10)

# Filepath label
filepath_label = Label(root, text="")
filepath_label.grid(row=2, column=0, padx=10, pady=10)

# Preprocess button
preprocess_btn = Button(root, text="Preprocess Image", command=preprocess_image)
preprocess_btn.grid(row=0, column=1, padx=10, pady=10)

# Panel to display preprocessed image
panel_preprocessed = Label(root)
panel_preprocessed.grid(row=1, column=1, padx=10, pady=10)

# Segment button
segment_btn = Button(root, text="Segment Image", command=segment_image)
segment_btn.grid(row=0, column=2, padx=10, pady=10)

# Panel to display segmented image
panel_segmented = Label(root)
panel_segmented.grid(row=1, column=2, padx=10, pady=10)

# Extract features button
features_btn = Button(root, text="Extract Features", command=extract_features)
features_btn.grid(row=0, column=3, padx=10, pady=10)

# Styling for feature labels
feature_label_style = {
    'bg': 'white',
    'fg': 'black',
    'font': ('Helvetica', 12, 'bold'),
    'padx': 10,
    'pady': 5,
    'relief': 'raised',
    'borderwidth': 2,
}

# Labels for extracted features
contrast_label = Label(root, text="Contrast: ", **feature_label_style)
contrast_label.grid(row=1, column=3, padx=10, pady=2, sticky="ew")
entropy_label = Label(root, text="Entropy: ", **feature_label_style)
entropy_label.grid(row=2, column=3, padx=10, pady=2, sticky="ew")
rms_label = Label(root, text="RMS: ", **feature_label_style)
rms_label.grid(row=3, column=3, padx=10, pady=2, sticky="ew")
variance_label = Label(root, text="Variance: ", **feature_label_style)
variance_label.grid(row=4, column=3, padx=10, pady=2, sticky="ew")
mean_label = Label(root, text="Mean: ", **feature_label_style)
mean_label.grid(row=5, column=3, padx=10, pady=2, sticky="ew")
energy_label = Label(root, text="Energy: ", **feature_label_style)
energy_label.grid(row=6, column=3, padx=10, pady=2, sticky="ew")
size_label = Label(root, text="Size: ", **feature_label_style)
size_label.grid(row=7, column=3, padx=10, pady=2, sticky="ew")

# Detect button
detect_btn = Button(root, text="Detection", command=predict_image)
detect_btn.grid(row=3, column=1, padx=10, pady=10)

# Labels for result and type
result_label = Label(root, text="Result: ", font=("Helvetica", 16))
result_label.grid(row=4, column=1, padx=10, pady=10)
type_label = Label(root, text="", font=("Helvetica", 14))
type_label.grid(row=5, column=1, padx=10, pady=10)

# Model accuracy label
# accuracy_label = Label(root, text="Model Accuracy: ")
# accuracy_label.grid(row=6, column=1, padx=10, pady=10)

# More Info button
info_btn = Button(root, text="For Doctor Contact Info / For More Info", command=open_link)
info_btn.grid(row=7, column=1, padx=10, pady=10)

root.mainloop()
