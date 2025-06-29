from src.dataloader import create_dataset

dataset, tokenizer = create_dataset()

for img_tensor, caption in dataset.take(1):
    print("🖼️ Image feature shape:", img_tensor.shape)
    print("📝 Caption shape:", caption.shape)
    break