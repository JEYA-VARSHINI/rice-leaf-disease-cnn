from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import json
import os


app = Flask(__name__)


# -----------------------------
# Load trained model
# -----------------------------
model = load_model("rice_leaf_mobilenetv2.keras")


# -----------------------------
# Load class names
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# -----------------------------
# Upload folder
# -----------------------------
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# Home route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":

        # Check whether file exists
        if "file" not in request.files:
            error = "Please select an image."
            return render_template(
                "index.html",
                error=error
            )

        file = request.files["file"]

        # Check filename
        if file.filename == "":
            error = "Please select an image."
            return render_template(
                "index.html",
                error=error
            )

        # Save uploaded image
        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(image_path)


        # -----------------------------
        # Image preprocessing
        # -----------------------------

        img = load_img(
            image_path,
            target_size=(128, 128)
        )

        img_array = img_to_array(img)

        # Same preprocessing used during training
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # -----------------------------
        # Prediction
        # -----------------------------

        print("STEP 1: Image loaded", flush=True)
        print("STEP 2: Starting model prediction...", flush=True)

        predictions = model.predict(img_array, verbose=0)

        print("STEP 3: Model prediction completed", flush=True)

        predicted_index = np.argmax(
            predictions[0]
        )

        prediction = class_names[
            predicted_index
        ]

        confidence = round(
            float(
                predictions[0][predicted_index]
            ) * 100,
            2
        )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


# -----------------------------
# Run Flask application
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)