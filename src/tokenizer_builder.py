import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

CAPTIONS_FILE = "data/captions_clean.csv"
TOKENIZER_PATH = "data/tokenizer.pkl"

print("🔤 Fitting tokenizer...")

# Load captions
df = pd.read_csv(CAPTIONS_FILE)
captions = df['caption'].astype(str).tolist()

# ✅ KEEP special characters like < and >
tokenizer = Tokenizer(oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~')  # removed '<' and '>' from filters
tokenizer.fit_on_texts(captions)

# Save tokenizer
with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer created and saved to", TOKENIZER_PATH)
