# 📧 Email Classifier with PII Masking (SBERT + XGBoost)

This project detects and masks PII (Personally Identifiable Information) from emails and classifies the email into categories like `Request`, `Incident`, `Problem`, or `Change`.

---

##  Features

- SBERT embeddings (`all-mpnet-base-v2`)
- XGBoost classifier for final prediction
- Custom PII detection (Aadhar, emails, phone, names, etc.)
- Masking with entity types
- Flask API + Gradio UI

---

##  Folder Structure

- `app_flask.py` – Flask backend (API only)
- `app.py` – Gradio UI (for Hugging Face)
- `api.py` – API route logic
- `models.py` – Model training, comparison, and saving
- `utils.py` – PII detection and masking
- 'Applying_PII_Masking'- Applying PII masking for Original data
- `sbert_encoder_model/` – SBERT model
- `sbert_final_model.pkl` – XGBoost classifier
- `label_encoder.pkl` – Label encoder

---

```bash
# Create environment and install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

# Run API server
python app.py

# OR run Gradio UI (for local testing)
python app_gradio.py
