import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array

# Paths
IMAGE_DIR = os.path.join("data", "images")
FEATURE_DIR = os.path.join("data", "features")

# Ensure output directory exists
os.makedirs(FEATURE_DIR, exist_ok=True)

def build_model():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
    return model

def extract_feature(img_path, model):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.squeeze()

def main():
    model = build_model()
    images = os.listdir(IMAGE_DIR)

    print("🔍 Extracting features from images...")

    for img_name in tqdm(images):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        feature = extract_feature(img_path, model)

        # Save as .npy with the same base filename
        feature_path = os.path.join(FEATURE_DIR, img_name.split('.')[0] + ".npy")
        np.save(feature_path, feature)

    print(f"✅ Features saved in {FEATURE_DIR}")


if __name__ == "__main__":
    main()
