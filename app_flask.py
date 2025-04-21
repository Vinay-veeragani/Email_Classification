from flask import Flask, request, Response
import joblib
from sentence_transformers import SentenceTransformer
import json
from api import classify  
from utils import detect_pii
app = Flask(__name__)

# 🔄 Load models
print("🔄 Loading models...")
sbert_model = SentenceTransformer("sbert_encoder_model")
classifier = joblib.load("sbert_final_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route('/classify', methods=['POST'])
def classify_route():
    data = request.get_json()
    email_text = data.get("email_text", "")

    if not email_text:
        return Response(json.dumps({"error": "email_text is required"}), status=400, mimetype='application/json')

    # ✅ Reuse shared function from api.py
    category, masked_email, original_email, formatted_entities = classify(
        email_text, sbert_model, classifier, label_encoder
    )

    # ✅ Strict output JSON format
    response = {
        "input_email_body": original_email,
        "list_of_masked_entities": [
            {
                "position": [ent["start"], ent["end"]],
                "classification": ent["type"],
                "entity": ent["value"]
            } for ent in detect_pii(email_text)
        ],
        "masked_email": masked_email,
        "category_of_the_email": category
    }

    return Response(json.dumps(response), mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
