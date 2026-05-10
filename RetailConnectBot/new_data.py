from datasets import load_dataset
import pandas as pd

ds = load_dataset("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset")
df = ds["train"].to_pandas()
print(df.columns)
print(df.head())
df.to_csv("retail_dataset.csv", index=False)
print(f"Saved {len(df)} rows")