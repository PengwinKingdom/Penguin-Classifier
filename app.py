from flask import Flask, request,jsonify,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from io import BytesIO

app=Flask(__name__)

model=load_model('model/penguin_classifier.h5')
class_names =['Adelie','African','Chinstrap','Emperor',
              'Erect Crested','Fiordland','Galapagos','Gentoo',
              'Humboldt','King','Little','Macaroni',
              'Magellanic','Northern Rockhopper','Royal','Snares',
              'Southern Rockhopper','Yellow Eyed']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'No se envio una imagen'}),400
    
    file =request.files['file']
    if file.filename=='':
        return jsonify({'error':'Nombre de archivo vacio'}),400
    
    try:
        #Cargar y procesar imagen
        target_size=(224,224)
        img=load_img(BytesIO(file.read()),target_size=target_size)
        img_array=img_to_array(img)
        img_array=img_array/255.0
        img_array=np.expand_dims(img_array,axis=0)
        
        #Prediccion
        predictions = model.predict(img_array)
        predicted_class=class_names[np.argmax(predictions[0])]
        
        return jsonify({'species':predicted_class})
    
    except Exception as e:
        return jsonify({'error':str(e)}),500

if __name__=='__main__':
    app.run(debug=True)            
    