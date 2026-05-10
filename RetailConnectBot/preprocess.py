import pandas as pd
import re
import json
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# STEP 1: Data Preprocessing
# ─────────────────────────────────────────

def clean_text(text):
    """
    Cleans raw user input:
    - Lowercase
    - Remove punctuation / special characters
    - Collapse extra whitespace
    """
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # keep letters, digits, spaces
    text = re.sub(r'\s+', ' ', text)           # remove extra spaces
    return text


def preprocess(input_path: str, output_dir: str = "."):
    # 1. Load dataset
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

    # 2. Clean input text
    df['clean_input'] = df['input'].apply(clean_text)

    # 3. Encode intent labels to integers
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['intent'])

    # 4. Save label mapping -> { 0: "greeting", 1: "price_inquiry", ... }
    label_map = {i: cls for i, cls in enumerate(le.classes_)}
    with open(f"{output_dir}/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved -> {label_map}")

    # 5. Save response map -> { "greeting": "Hello! ...", ... }
    response_map = df.groupby('intent')['response'].first().to_dict()
    with open(f"{output_dir}/response_map.json", 'w') as f:
        json.dump(response_map, f, indent=2)
    print(f"Response map saved for {len(response_map)} intents")

    # 6. Save cleaned dataset
    out_path = f"{output_dir}/cleaned_data.csv"
    df[['clean_input', 'label', 'intent']].to_csv(out_path, index=False)
    print(f"Cleaned data saved -> {out_path}")

    # 7. Show sample transformations
    print("\nSample transformations:")
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"  ORIGINAL : {row['input']}")
        print(f"  CLEANED  : {row['clean_input']}")
        print()

    return df, le


# ─────────────────────────────────────────
if __name__ == "__main__":
    preprocess(
        input_path="llm_level_dataset (1).csv",
        output_dir="."
    )
