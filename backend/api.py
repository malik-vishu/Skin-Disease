from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
CORS(app)

from tensorflow.keras.models import load_model
# model_path = 'your_model.h5'
loaded_model = tf.keras.models.load_model('D:/Projects/Sem-6/backend/d1.h5')

classes = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
           'Atopic Dermatitis Photos', 'Cellulitis Impetigo and other Bacterial Infections',
           'Eczema Photos', 'Exanthems and Drug Eruptions', 'Herpes HPV and other STDs Photos',
           'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
           'Melanoma Skin Cancer Nevi and Moles', 'Poison Ivy Photos and other Contact Dermatitis',
           'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors',
           'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections',
           'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']


@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('files')
    predictions = []
    
    for file in files:
        if file:
            filename = file.filename
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            
            img = load_img(filepath, target_size=(192, 192))
            x = img_to_array(img)
            x /= 255
            x = np.expand_dims(x, axis=0)

            pred = loaded_model.predict(x)
            ans = np.argmax(pred[0])
            result = classes[ans]

            predictions.append({'filename': filename, 'prediction': result})
    
    return jsonify(predictions)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)




