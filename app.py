from flask import Flask, request,jsonify,render_template,redirect,url_for,session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from io import BytesIO
from werkzeug.utils import secure_filename
import gdown

app=Flask(__name__)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "penguin_classifier.h5")
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/file/d/1q4kJQPJf0-0jRyv2wfYV7XCjMxYLX-OT")

def ensure_model_downloaded():
    """Download the model file if it does not exist locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return

    if not MODEL_URL or MODEL_URL == "https://drive.google.com/file/d/1q4kJQPJf0-0jRyv2wfYV7XCjMxYLX-OT":
        raise RuntimeError(
            "MODEL_URL is not set. Please configure a valid link to the model file."
        )

    try:
        print(f"[model] Downloading from: {MODEL_URL}")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False,fuzzy=True)
    except Exception as e:
        raise RuntimeError(f"Model download failed: {e}")


ensure_model_downloaded()

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

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename=='':
        return redirect(url_for('index'))
    
    file =request.files['file']
    
    try:
        file_bytes=file.read()
        
        filename=secure_filename(file.filename)
        file_path=os.path.join('static/uploads',filename)
        
        os.makedirs('static/uploads',exist_ok=True)
        with open(file_path,'wb') as f:
            f.write(file_bytes)
        
        # Load and preprocess the uploaded image
        target_size=(224,224)
        img=load_img(file_path,target_size=target_size)
        img_array=img_to_array(img)
        img_array=img_array/255.0
        img_array=np.expand_dims(img_array,axis=0)
        
        
        # Perform prediction
        probs=model.predict(img_array)[0].astype(float)
        
        # Sort predictions by probability (highest first)
        sorted_idx = np.argsort(probs)[::-1]
        top1_idx = int(sorted_idx[0])
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        
        
        # Confidence thresholds
        threshold = 0.85
        margin=0.20
        
        if (top1_prob<threshold) or ((top1_prob - top2_prob) < margin):
            return render_template(
                'result.html',
                species="This is not a penguin",
                confidence=round(top1_prob*100,2),
                image_url=file_path
            )
        predicted_class=class_names[top1_idx]
        confidence=round(top1_prob*100,2)
        return render_template('result.html',species=predicted_class,confidence=confidence,image_url=file_path)
    
    except Exception as e:
        return jsonify({'error':str(e)}),500


if __name__=='__main__':
    app.run(debug=True)            
    