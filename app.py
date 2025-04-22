# app_gradio.py
import gradio as gr
import joblib
from sentence_transformers import SentenceTransformer
from api import classify  

# Load models
sbert_model = SentenceTransformer("sbert_encoder_model")
classifier = joblib.load("sbert_final_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

iface = gr.Interface(
    fn=lambda text: classify(text, sbert_model, classifier, label_encoder),
    inputs=gr.Textbox(
        lines=10,
        label="📨 Enter your email",
        placeholder="Paste your email content here..."
    ),
    outputs=[
        gr.JSON(label="📦 Strict API JSON Output"),
        gr.Textbox(label="📝 Original Email Body"),
        gr.Dataframe(headers=["Type", "Entity", "Position"], label="🛡️ Detected Entities"),
        gr.Textbox(label="🔒 Masked Email"),
        gr.Label(label="📬 Predicted Category")
        
    ],
    title="Email Classification for Support Team",
    description="This app masks PII and classifies emails using SBERT + XGBoost",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
