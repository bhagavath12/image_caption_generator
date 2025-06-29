import pandas as pd

path = "data/captions_clean.csv"
df = pd.read_csv(path)

# Force update to proper format
df['caption'] = df['caption'].apply(lambda x: f"<start> {x.strip().replace('<start>', '').replace('<end>', '')} <end>")

df.to_csv(path, index=False)
print(" Captions file forcibly cleaned with proper <start> and <end>")
