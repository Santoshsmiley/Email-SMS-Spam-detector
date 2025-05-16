# app.py
from flask import Flask, request, jsonify
import re
from flask_cors import CORS
import pickle



app = Flask(__name__)
CORS(app)

# Load model and vectorizer
def load_model_vectorizer():
    with open('../models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_vectorizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/predict', methods=['POST'])
def predict_email():
    data = request.get_json()
    email_text = data.get('email_text', '')
    
    if not email_text.strip():
        return jsonify({'error': 'Please enter some email text.'}), 400
    
    cleaned = clean_text(email_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    result = "Spam" if pred[0] == 1 else "Not Spam"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)