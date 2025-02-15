from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from mtcnn.mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray

app = Flask(__name__)
DATA_DIR = "image_classification_dataset"  # Directory for training images
UPLOAD_DIR = "uploads"  # Directory for uploaded images to classify
MODEL_PATH = "person_classifier_model.h5"
CLASS_NAMES_PATH = "class_names.npy"
detector = MTCNN()

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model and class names
model = models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
class_names = np.load(CLASS_NAMES_PATH).tolist() if os.path.exists(CLASS_NAMES_PATH) else []

def extract_face_from_image(image_path, required_size=(128, 128)):
    image = plt.imread(image_path)
    faces = detector.detect_faces(image)
    face_images = []
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)
    return face_images

@app.route('/')
def index():
        return render_template('index.html')

@app.route('/upload_train', methods=['GET', 'POST'])
def upload_train():
    if request.method == 'POST':
        class_name = request.form['class_name']
        files = request.files.getlist('images')
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for file in files:
            file.save(os.path.join(class_dir, file.filename))
        
        # Train model immediately after upload
        global class_names, model
        image_data, labels, loaded_class_names = load_data(DATA_DIR)
        class_names = loaded_class_names  # Update global class names

        # Convert to NumPy arrays
        image_data_np = image_data.numpy()
        labels_np = labels.numpy()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(image_data_np, labels_np, test_size=0.2, random_state=42)

        # Create, compile, and train model
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)

        # Save model and class names
        model.save(MODEL_PATH)
        np.save(CLASS_NAMES_PATH, class_names)
        
        return redirect(url_for('training_success'))
    return render_template('upload_train.html')

@app.route('/training_success')
def training_success():
    return render_template('training_success.html')

@app.route('/train', methods=['GET'])
def train_model():
    global class_names, model
    image_data, labels, loaded_class_names = load_data(DATA_DIR)
    class_names = loaded_class_names  # Update global class names

    # Convert to NumPy arrays
    image_data_np = image_data.numpy()
    labels_np = labels.numpy()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(image_data_np, labels_np, test_size=0.2, random_state=42)

    # Create, compile, and train model
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)

    # Save model and class names
    model.save(MODEL_PATH)
    np.save(CLASS_NAMES_PATH, class_names)
    return redirect(url_for('classify'))

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save to upload directory
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(file_path)

            # Extract face and predict
            face_images = extract_face_from_image(file_path)
            if face_images:
                image_array = np.expand_dims(face_images[0], axis=0) / 255.0
                prediction = model.predict(image_array)
                predicted_index = np.argmax(prediction)
                # Ensure index is within valid range
                if predicted_index < len(class_names):
                    predicted_class = class_names[predicted_index]
                else:
                    predicted_class = "Unknown Class"
                return render_template('result.html', predicted_class=predicted_class)
    return render_template('classify.html')

@app.route('/trained_models')
def trained_models():
    trained_models_list = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]
    return render_template('trained_models.html', models=trained_models_list)

@app.route('/delete_model/<model_name>', methods=['POST'])
def delete_model(model_name):
    model_dir = os.path.join(DATA_DIR, model_name)
    if os.path.exists(model_dir):
        # Remove all files in the directory first
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        # Remove the directory
        os.rmdir(model_dir)
    return redirect(url_for('trained_models'))

def load_data(data_dir):
    image_data = []
    labels = []
    loaded_class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    for idx, class_name in enumerate(loaded_class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            extracted_face = extract_face_from_image(img_path)
            if extracted_face:
                img_array = tf.keras.utils.img_to_array(extracted_face[0])
                image_data.append(img_array)
                labels.append(idx)
    # Convert to tensors
    image_data = tf.convert_to_tensor(image_data) / 255.0
    labels = tf.convert_to_tensor(labels)
    return image_data, labels, loaded_class_names

def create_model():
    return models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')  # Correct number of classes
    ])

if __name__ == '__main__':
    app.run(debug=True)