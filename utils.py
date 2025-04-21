import re
import spacy
import pandas as pd

nlp = spacy.load('en_core_web_md')

NON_PERSON_KEYWORDS = {'aadhar', "Aadhar",'expiry', 'cvv', 'dob', 'card', 'contact',"Card"}



def detect_pii(text):
    entities = []

     # Credit/Debit Cards (16 digits with optional separators)
    card_regex = r'\b(?:\d{4}[- ]?){3}\d{4}\b'
    for match in re.finditer(card_regex, text):
        entities.append({
            "start": match.start(),
            "end": match.end(),
            "type": "credit_debit_no",
            "value": match.group()
        })

    # CVV Numbers (with context)
    cvv_regex = r'(?i)\b(?:CVV|CVC)\b\s*[:]?\s*(\d{3,4})\b' 
    for match in re.finditer(cvv_regex, text):
        entities.append({
            "start": match.start(1),
            "end": match.end(1),
            "type": "cvv_no",
            "value": match.group(1)
        })


    # Detect Aadhar FIRST (priority)
    aadhar_regex = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}\b'  # Matches "1234-5678-9012"
    for match in re.finditer(aadhar_regex, text):
        entities.append({
            "start": match.start(),
            "end": match.end(),
            "type": "aadhar_num",
            "value": match.group()
        })
    
    #Phone numbers (exclude 12-digit sequences)
    phone_regex = r'(\+\d{1,3}[- ]?)?\(?\d{1,4}\)?[-.\s]?(?:\d{3,4}[-.\s]?){1,2}\d{3,4}\b(?!\d{8})'
    for match in re.finditer(phone_regex, text):
        entities.append({
            "start": match.start(),
            "end": match.end(),
            "type": "phone_number",
            "value": match.group()
        })
    
    #Detect Full Names with spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if  ent.label_ == "PERSON" and ent.text.lower() not in NON_PERSON_KEYWORDS:
            entities.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "type": "full_name",
                "value": ent.text
            })
    
    # Email Addresses
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_regex, text, re.IGNORECASE):
        entities.append({
            "start": match.start(),
            "end": match.end(),
            "type": "email",
            "value": match.group()
        })
    
    # Date of Birth
    dob_regex = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b'
    for match in re.finditer(dob_regex, text):
        entities.append({
            "start": match.start(),
            "end": match.end(),
            "type": "dob",
            "value": match.group()
        })
    
    
    
    # Expiry Dates
    expiry_regex = r'\b(0[1-9]|1[0-2])/(\d{2}|\d{4})\b'
    for match in re.finditer(expiry_regex, text):
        entities.append({
            "start": match.start(),
            "end": match.end(),
            "type": "expiry_no",
            "value": match.group()
        })
    
    # Resolve entity overlaps (keep longest matches)
    entities.sort(key=lambda x: x['start'])
    filtered_entities = []
    prev_end = 0
    
    for entity in entities:
        if entity['start'] >= prev_end:
            filtered_entities.append(entity)
            prev_end = entity['end']
        else:
            current_length = entity['end'] - entity['start']
            existing_length = filtered_entities[-1]['end'] - filtered_entities[-1]['start']
            if current_length > existing_length:
                filtered_entities.pop()
                filtered_entities.append(entity)
                prev_end = entity['end']
    
    return filtered_entities



def mask_email(text, entities):
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    masked_text = text
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        masked_text = masked_text[:start] + f"[{entity['type']}]" + masked_text[end:]
    return masked_text

#Step 2 --- Applying PII masking to Original Dataset and Saving 
