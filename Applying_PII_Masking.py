import pandas as pd
from utils import detect_pii,mask_email
# Loading  and cleaning dataset
df = pd.read_csv(r"C:\Users\veera\Downloads\combined_emails_with_natural_pii.csv")
df.dropna(subset=['email', 'type'], inplace=True)
df.drop_duplicates(inplace=True)
df['email'] = df['email'].str.lower()

# Apply detect + mask
def generate_masked_email(text):
    entities = detect_pii(text)
    return mask_email(text, entities)

# Applying  to all rows
df['masked_email'] = df['email'].apply(generate_masked_email)

# Saving only what's needed for training
df[['masked_email', 'type']].to_csv("email_masked_for_training.csv", index=False)

print("Masked email column generated and saved successfully!")
