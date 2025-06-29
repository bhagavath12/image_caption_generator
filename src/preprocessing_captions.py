import pandas as pd

INPUT_PATH = "data/captions_clean.csv"
OUTPUT_PATH = "data/captions_clean.csv"  # Overwrites the original file

df = pd.read_csv(INPUT_PATH)

# Only add tokens if they don't already exist (safe check for re-runs)
if not df['caption'].iloc[0].startswith("<start>"):
    df['caption'] = df['caption'].apply(lambda x: f"<start> {x.strip()} <end>")
    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Captions updated with <start> and <end> tokens.")
else:
    print("⚠️ Captions already contain <start> and <end> — no changes made.")
