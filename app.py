from flask import Flask, render_template, request
import joblib
import torch
import os
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from text_preprocessing import clean_text
from stylometric_extraction import extract_stylometric_features

app = Flask(__name__)

# --- Load Models and Tokenizers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLO_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'stylometric_model.pkl')
TRANS_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'transformer_classifier.pkl')

print("Loading models...")
stylo_model = joblib.load(STYLO_MODEL_PATH)
trans_classifier = joblib.load(TRANS_MODEL_PATH)

print("Loading transformer backbone (DistilBERT)...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
trans_backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
trans_backbone.eval()

def get_prediction_label(pred):
    return "AI-generated" if pred == 1 else "Human-written"

def get_explanation(label):
    if label == "AI-generated":
        return "This text appears highly structured and consistent, which is typical of AI-generated content."
    else:
        return "This text shows natural variation and informal patterns typical of human writing."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        
        if input_text:
            # 1. Preprocess text (for both models)
            cleaned_text = clean_text(input_text)
            
            # 2. Stylometric Prediction
            stylo_features_dict = extract_stylometric_features(input_text)
            stylo_features = [list(stylo_features_dict.values())]
            
            # Get probabilities
            stylo_proba = stylo_model.predict_proba(stylo_features)[0]
            stylo_pred = int(np.argmax(stylo_proba))
            stylo_conf = float(stylo_proba[stylo_pred])
            stylo_label = get_prediction_label(stylo_pred)
            
            # 3. Transformer Prediction
            with torch.no_grad():
                inputs = tokenizer(cleaned_text, padding=True, truncation=True, max_length=128, return_tensors='pt')
                outputs = trans_backbone(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
                
                # Get probabilities
                trans_proba = trans_classifier.predict_proba(cls_emb)[0]
                trans_pred = int(np.argmax(trans_proba))
                trans_conf = float(trans_proba[trans_pred])
                trans_label = get_prediction_label(trans_pred)
            
            # 4. Hybrid Decision Logic
            final_prediction = ""
            status_note = ""
            explanation = ""
            final_conf = 0.0
            
            if stylo_label == trans_label:
                final_prediction = stylo_label
                explanation = get_explanation(final_prediction)
                final_conf = (stylo_conf + trans_conf) / 2
            else:
                final_prediction = "Mixed Signals"
                status_note = "Mixed signals detected: The text contains characteristics of both human and AI writing styles."
                # No average confidence for mixed signals, handled in UI
            
            result = {
                "stylo": stylo_label,
                "stylo_conf": round(stylo_conf * 100, 1),
                "trans": trans_label,
                "trans_conf": round(trans_conf * 100, 1),
                "final": final_prediction,
                "final_conf": round(final_conf * 100, 1) if final_conf > 0 else None,
                "note": status_note,
                "explanation": explanation,
                "stylo_raw": stylo_features_dict
            }

    return render_template('index.html', text=input_text, result=result)

if __name__ == '__main__':
    app.run(debug=True)
