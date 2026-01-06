from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import base64
from PIL import Image
import io

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "BurntSkinClassifier.h5")

model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)
class_names = ['first degree', 'second degree', 'third degree']

# Recomendaciones
recomendaciones = {
    "No hay quemadura": "La imagen analizada no sugiere quemadura.",
    "first degree": "Aplicar agua fría 10-15 min. No usar hielo. Aloe vera.",
    "second degree": "No revientes las ampollas. Lava suavemente y cubre con gasas.",
    "third degree": "Emergencia. Busca atención médica inmediata."
}

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def aplicar_reglas(predicted_class, confidence):
    """
    Reglas solicitadas:
    - first degree, <40% → No hay quemadura
    - second degree, <40% → first degree
    - third degree, <40% → second degree
    """

    if predicted_class == "first degree" and confidence < 40:
        return "No hay quemadura"


    return predicted_class


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "Empty file", 400

    STATIC_DIR = os.path.join(BASE_DIR, "static")
    img_path = os.path.join(STATIC_DIR, "uploaded.jpg")

    file.save(img_path)

    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = float(np.max(score)) * 100

    #  Aplicar reglas nuevas
    predicted_class = aplicar_reglas(predicted_class, confidence)

    recomendacion = recomendaciones.get(predicted_class, "No hay recomendación disponible.")

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=f"{confidence:.2f}%",
        recomendacion=recomendacion,
        image_url="uploaded.jpg"
    )


@app.route("/predict_cam", methods=["POST"])
def predict_cam():
    data = request.get_json()
    img_base64 = data["image"].split(",")[1]

    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize(IMG_SIZE)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = float(np.max(score)) * 100

    # Aplicar reglas nuevas también aquí
    predicted_class = aplicar_reglas(predicted_class, confidence)

    recomendacion = recomendaciones.get(predicted_class, "No hay recomendación disponible.")

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%",
        "recomendacion": recomendacion
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

