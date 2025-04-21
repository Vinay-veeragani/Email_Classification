import pandas as pd
import joblib
import time
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load and encode
df = pd.read_csv("email_masked_for_training.csv")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['type'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['masked_email'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# Load SBERT model
print("Loading SBERT model...")
sbert_model = SentenceTransformer("all-mpnet-base-v2")

# Generate embeddings 
print(" Generating embeddings...")
X_train_embed = sbert_model.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
X_test_embed = sbert_model.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)

# Dictionary of models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = {}

# Train & evaluate all models
for name, model in models.items():
    print(f"\n Training {name}...")
    start = time.time()
    model.fit(X_train_embed, y_train)
    end = time.time()

    y_pred = model.predict(X_test_embed)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n Classification Report for {name}:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "train_time": round(end - start, 2)
    }

# Comparing the  results
print("\n🧠 Model Comparison Summary:")
for name, info in results.items():
    print(f"{name:<20} | Acc: {info['accuracy']:.4f} | F1: {info['f1_score']:.4f} | Time: {info['train_time']}s")

# Saving  the best model 
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']
joblib.dump(best_model, "sbert_final_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
sbert_model.save("sbert_encoder_model")

print(f"\nBest Model Saved: {best_model_name}")