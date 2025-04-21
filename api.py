# Detect and PII masking 
from utils import detect_pii, mask_email
def classify(email_text, sbert_model, classifier, label_encoder):
    if not email_text.strip():
        return "", [], "", "Email is required.", {}

    entities = detect_pii(email_text)
    masked_email = mask_email(email_text, entities)
    embedding = sbert_model.encode([masked_email])
    pred_index = classifier.predict(embedding)[0]
    category = label_encoder.inverse_transform([pred_index])[0]

    formatted_entities = [
        [ent["type"], ent["value"], f"{ent['start']}-{ent['end']}"]
        for ent in entities
    ]

    # Build strict JSON response
    json_response = {
        "input_email_body": email_text,
        "list_of_masked_entities": [
            {
                "position": [ent["start"], ent["end"]],
                "classification": ent["type"],
                "entity": ent["value"]
            } for ent in entities
        ],
        "masked_email": masked_email,
        "category_of_the_email": category
    }

    return email_text, formatted_entities, masked_email, category, json_response
