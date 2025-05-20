from diffusers import StableDiffusion3Pipeline
from peft import get_peft_model, LoraConfig
import torch
from torch.optim import AdamW
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from transformers import T5Tokenizer
import pandas as pd
import torch.utils.data
import os

# Verify the new cache path
print("Cache directory: ", os.getenv('HF_HOME'))

# Error handling for model loading
try:
    print("Loading the Stable Diffusion 3.5 Large Turbo model...")
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.float32)
    pipe.to("cuda")  # Move the model to GPU
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Error handling for tokenizer loading
try:
    print("Loading T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=True)  # Use fast tokenizer for better performance
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding works correctly
    print(f"Tokenizer type: {type(tokenizer)}")
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# LoRA: Apply Low-Rank Adaptation (LoRA) for lightweight model tuning
try:
    print("Applying LoRA adaptation to the UNet...")
    lora_config = LoraConfig(r=4, lora_alpha=32, lora_dropout=0.1)
    lora_model = get_peft_model(pipe.unet, lora_config)  # Apply LoRA to the UNet part
    print("LoRA applied successfully!")
except Exception as e:
    print(f"Error applying LoRA: {e}")
    exit(1)

# DataLoader and transformations for image dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return None, None

# Error handling for CSV file loading
try:
    print("Loading training data from CSV...")
    data = pd.read_csv('photo_labels.csv')
    image_paths = data['image_path'].tolist()
    labels = data['label'].tolist()
    print(f"Loaded {len(image_paths)} images and {len(labels)} labels.")
except Exception as e:
    print(f"Error loading CSV data: {e}")
    exit(1)

# Transformation and dataset creation
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
dataset = CustomDataset(image_paths, labels, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# AdamW optimizer for model training
optimizer = AdamW(lora_model.parameters(), lr=5e-6)

# Initialize Accelerator for parallel training (if using multiple GPUs)
accelerator = Accelerator()

# Training loop with error handling
epochs = 5  # Define number of epochs
for epoch in range(epochs):
    lora_model.train()
    for images, labels in dataloader:
        if images is None or labels is None:
            continue  # Skip this batch if there's an error with the image

        optimizer.zero_grad()

        try:
            # Move images to GPU
            images = images.to("cuda")

            # Tokenize the labels (since we are using T5 tokenizer for prompts)
            tokenized_labels = tokenizer(labels, padding=True, return_tensors="pt").to("cuda")
            input_ids = tokenized_labels["input_ids"]

            # Run the model with LoRA adaptation
            outputs = pipe(images, prompt=input_ids)  # Pass tokenized input ids as the prompt

            # Compute loss and backpropagate
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        except Exception as e:
            print(f"Error during training at epoch {epoch+1}: {e}")
            continue

# Save the fine-tuned model
try:
    print("Saving the fine-tuned model...")
    pipe.save_pretrained("fine_tuned_model")
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving the model: {e}")
    exit(1)

# Generate an image with the fine-tuned model
try:
    print("Generating image with the fine-tuned model...")
    prompt = "<fkylmz> stands on a mountain, sunset background"
    generated_image = pipe(prompt).images[0]

    # Show and save the generated image
    generated_image.show()
    generated_image.save("generated_image.png")
    print("Image generated and saved successfully!")

except Exception as e:
    print(f"Error during image generation: {e}")
