import pandas as pd
import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# STEP 2: TF-IDF Tokenization & Encoding
# ─────────────────────────────────────────
# TF-IDF (Term Frequency - Inverse Document Frequency):
#   - TF  : how often a word appears in THIS sentence
#   - IDF : penalizes words common across ALL sentences (like "the", "is")
#   - Result: each sentence becomes a numeric vector
#
# Example:
#   "what is the price" -> [0, 0.8, 0, 0.6, 0, ...]  (sparse vector)
#   "hey price??"       -> [0, 0.9, 0, 0,   0, ...]
# ─────────────────────────────────────────

def build_vectorizer(data_path: str, output_dir: str = "."):
    # 1. Load cleaned data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")

    X = df['clean_input'].values
    y = df['label'].values

    # 2. Train/Test split (80/20, stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # ensures each intent is proportionally represented
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Fit TF-IDF on training data only (NEVER fit on test data)
    vectorizer = TfidfVectorizer(
        max_features=3000,      # top 3000 words by TF-IDF score
        ngram_range=(1, 2),     # unigrams + bigrams: "price" AND "what price"
        min_df=2,               # ignore words appearing in < 2 documents
        sublinear_tf=True       # apply log(TF) to reduce impact of very frequent words
    )

    X_train_vec = vectorizer.fit_transform(X_train).toarray()  # fit+transform train
    X_test_vec  = vectorizer.transform(X_test).toarray()       # transform only test

    print(f"Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"Feature matrix  : {X_train_vec.shape}  (samples x features)")

    # 4. Save vectorizer for reuse during inference
    with open(f"{output_dir}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved -> tfidf_vectorizer.pkl")

    # 5. Save splits for training step
    np.save(f"{output_dir}/X_train.npy", X_train_vec)
    np.save(f"{output_dir}/X_test.npy",  X_test_vec)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_test.npy",  y_test)
    print("Train/test arrays saved as .npy files")

    # 6. Show a sample encoding
    sample = "hey whats the price for jeans??"
    sample_clean = sample.lower()
    import re
    sample_clean = re.sub(r'[^a-z0-9\s]', '', sample_clean).strip()
    sample_vec = vectorizer.transform([sample_clean]).toarray()
    nonzero = np.count_nonzero(sample_vec)
    print(f"\nSample encoding:")
    print(f"  Input   : '{sample}'")
    print(f"  Cleaned : '{sample_clean}'")
    print(f"  Vector  : shape={sample_vec.shape}, non-zero features={nonzero}")

    return vectorizer, X_train_vec, X_test_vec, y_train, y_test


# ─────────────────────────────────────────
if __name__ == "__main__":
    build_vectorizer(
        data_path="cleaned_data.csv",
        output_dir="."
    )
