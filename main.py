from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
import PIL
app = Flask(__name__)

CLASS_LABELS = [
    'Clams',
    'Corals',
    'Crabs',
    'Dolphin',
    'Eel',
    'Fish',
    'Jelly Fish',
    'Lobster',
    'Nudibranchs',
    'Octopus',
    'Otter',
    'Penguin',
    'Puffers',
    'Sea Rays',
    'Sea Urchins',
    'Seahorse',
    'Seal',
    'Sharks',
    'Shrimp',
    'Squid',
    'Starfish',
    'Turtle_Tortoise',
    'Whale'
    ]
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'models/model.keras'  
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    #Dummy Model for testing the app
    model  =    Sequential([
        Dense(10, input_shape=(224, 224, 3), activation='relu'),
        Dense(3, activation='softmax')  # Assuming a 3-class classification
    ])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_path, target_size=(224, 224)):  
    image = load_img(file_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  
    return image


@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return "<p>No file part in the request.</p>"

        file = request.files["file"]

        
        if file.filename == "":
            return "<p>No selected file.</p>"

    
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

     
            image = preprocess_image(file_path)
            predictions = model.predict(image)
            score = tf.nn.softmax(predictions[0])
            predicted_label = CLASS_LABELS[np.argmax(score)]
            result = f"Predicted class: {predicted_label}"

            return f"""
            <p>{result}</p>
            <p><a href="/">Upload another image</a></p>
            """

        return "<p>Invalid file type. Only JPG and PNG files are allowed.</p>"

    # Render the upload form
    return render_template_string("""
    <!doctype html>
    <title>Image Classification</title>
    <h1>Upload an Image for Prediction</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
        <input type="submit" value="Upload and Predict">
    </form>
    """)

if __name__ == "__main__":
    app.run(debug=True)