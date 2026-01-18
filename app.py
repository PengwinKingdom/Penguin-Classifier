from flask import Flask, request,jsonify,render_template,redirect,url_for,session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from io import BytesIO
from werkzeug.utils import secure_filename
import gdown
import base64
from PIL import Image


app=Flask(__name__)

# Limita tamaño de subida
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4MB

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "penguin_classifier.h5")
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/file/d/1q4kJQPJf0-0jRyv2wfYV7XCjMxYLX-OT")

def ensure_model_downloaded():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is not set. Please configure a valid link to the model file.")

    print(f"[model] Downloading from: {MODEL_URL}")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

# Descarga una vez en el arranque (no por request)
ensure_model_downloaded()

# Carga una vez en el arranque (no por request)
model=load_model(MODEL_PATH)

# List of penguin species supported by the model
class_names =['Adelie','African','Chinstrap','Emperor',
              'Erect Crested','Fiordland','Galapagos','Gentoo',
              'Humboldt','King','Little','Macaroni',
              'Magellanic','Northern Rockhopper','Royal','Snares',
              'Southern Rockhopper','Yellow Eyed']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']

    try:
        file_bytes = file.read()
        
        img_b64 = base64.b64encode(file_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{img_b64}"

        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        img = img.resize((224, 224))

        img_array = img_to_array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediccion
        probs = model.predict(img_array, verbose=0)[0].astype(float)

        sorted_idx = np.argsort(probs)[::-1]
        top1_idx = int(sorted_idx[0])
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0

        threshold = 0.85
        margin = 0.20

        if (top1_prob < threshold) or ((top1_prob - top2_prob) < margin):
            return render_template(
                "result.html",
                species="This is not a penguin",
                confidence=round(top1_prob * 100, 2),
                image_url=image_data_url
            )


        predicted_class = class_names[top1_idx]
        confidence = round(top1_prob * 100, 2)

        return render_template(
            'result.html',
            species=predicted_class,
            confidence=confidence,
            image_url=image_data_url
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__=='__main__':
    app.run(debug=True)            
    
