# 🖼️ Image Caption Generator

Generate natural language descriptions for images using a deep learning model with an Encoder-Decoder architecture (CNN-RNN).

---

## 📌 Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Project Structure](#project-structure)
* [How It Works](#how-it-works)
* [Setup Instructions](#setup-instructions)
* [Training](#training)
* [Inference](#inference)
* [Evaluation](#evaluation)
* [Sample Output](#sample-output)
* [Dependencies](#dependencies)
* [Data Management](#data-management)

---

## 🧠 Overview

This project takes an image as input and outputs a relevant caption describing the contents of the image. It uses:

* **CNN** (ResNet50 or InceptionV3) for feature extraction (Encoder)
* **LSTM** for generating captions (Decoder)
* **Tokenizer** to map words to integer sequences
* **Custom DataLoader** using TensorFlow's `tf.data`

---

## 🏗️ Architecture

### Encoder (CNN)

* Pre-trained CNN (e.g. InceptionV3)
* Removes final classification layer
* Outputs a 2048-dim feature vector

### Decoder (RNN)

* Embedding Layer
* LSTM with Attention
* Fully Connected Layer to predict vocabulary word at each time step

---

## 📁 Project Structure

```
image_caption_generator/
├── data/                         # Not pushed to GitHub (see Data Management)
│   ├── captions.txt              # Raw MSCOCO-style captions
│   ├── captions_clean.csv        # Preprocessed captions with <start> and <end>
│   ├── tokenizer.pkl             # Saved tokenizer
│   └── features/                 # Pre-extracted .npy feature files
│
├── src/
│   ├── encoder.py
│   ├── decoder.py
│   ├── dataloader.py
│   ├── train.py
│   ├── utils.py
│   ├── tokenizer_builder.py
│   ├── config.py
│   ├── extract_features.py
│   └── preprocessing_captions.py
│
├── inference.py
├── inference_utils.py
├── evaluate.py
├── test.py
├── test_dataloader.py
├── check_tokenizer.py
├── force_fix_captions.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. **Preprocess captions:** Clean and add `<start>` / `<end>` tokens
2. **Build tokenizer:** Fit on captions and save `tokenizer.pkl`
3. **Extract image features:** Use InceptionV3 to save `.npy` files
4. **Load Dataset:** Map captions to image features
5. **Train model:** Pass image + previous words into encoder/decoder
6. **Evaluate or infer:** Decode predicted words into caption

---

## 🧪 Setup Instructions

1. **Clone repo**

```bash
git clone https://github.com/bhagavath12/image_caption_generator.git
cd image_caption_generator
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Preprocess Captions**

```bash
python src/preprocessing_captions.py
# OR to forcibly fix missing tokens:
python force_fix_captions.py
```

4. **Build Tokenizer**

```bash
python src/tokenizer_builder.py
```

5. **Extract Image Features**

```bash
python src/extract_features.py
```

6. **Test Dataloader**

```bash
python test_dataloader.py
```

---

## 🏋️ Training

```bash
python src/train.py
```

Checkpoints will be saved to:

```
checkpoints/train/
```

---

## 🧠 Inference

```bash
python inference.py --image test.jpg
```

Or use the test wrapper:

```bash
python test.py
```

Make sure `test.jpg` is placed in the root directory.

---

## 📊 Evaluation

```bash
python evaluate.py
```

Returns BLEU score and optionally attention visualization.

---

## 🖼️ Sample Output

```
Input: 🐶 test.jpg
Output: "A brown dog is running through the grass."
```

---

## 📦 Dependencies

```txt
tensorflow>=2.10
numpy
pandas
matplotlib
pickle
opencv-python
scikit-learn
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📁 Data Management

The `data/` folder is **intentionally excluded** from GitHub to avoid:

* Uploading large `.npy` feature files
* Uploading raw image datasets
* Exceeding GitHub’s 100MB file limit

You must manually create the `data/` folder:

```
data/
├── captions.txt
├── captions_clean.csv
├── tokenizer.pkl
├── features/         # Contains *.npy files per image
└── images/           # Optional: original images
```

Ensure you generate required files using:

* `force_fix_captions.py`
* `tokenizer_builder.py`
* `extract_features.py`

---

## 💡 Notes

* BLEU scores may vary based on beam search, epochs, and data size
* Add `beam search` for better inference performance
* Preprocessing and tokenizer consistency are crucial

---

## 📮 Credits

* MSCOCO Dataset
* TensorFlow Tutorials
* Show, Attend and Tell (Xu et al.)

---

## 🛠️ Improvements You Can Add

* 🔍 Attention Visualization
* 🧠 Beam Search Decoding
* 📦 PyTorch version
* 📈 Gradio/Web demo

---

## 📬 License

MIT License
