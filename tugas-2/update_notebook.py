
import json
import os

notebook_path = "peft-food-recommendation.ipynb"

# Define the new logic as standard python strings
# We will insert these into the notebook cells.


code_install = """# 1. Install once per environment
# Pin versions to ensure compatibility and avoid 'modeling_layers' error
%pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
%pip install -q transformers==4.46.0 peft==0.13.2 datasets==3.1.0 accelerate==1.1.0 pandas
"""


code_imports = """import torch
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd

# 2. Setup Device
def pick_device():
    if hasattr(torch, "accelerator") and torch.accelerator.current_accelerator():
        return torch.accelerator.current_accelerator().type
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = pick_device()
print(f"Using device: {device}")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model
DATA_PATH = "../tugas-1/gofood_dataset.csv"       # Dataset path
OUTPUT_DIR = "tinyllama-gofood-lora"
"""

code_dataset = """# 3. Load and format dataset
def format_example(row):
    # Ensure columns exist, handle missing usage gracefully
    product = row.get('product', 'Makanan')
    # Handle potential float/int prices
    price = str(row.get('price', '0'))
    category = row.get('category', 'Umum')
    merchant_name = row.get('merchant_name', 'Merchant')
    merchant_area = row.get('merchant_area', 'Area')
    
    products_str = f"{product} (Rp{price})"
    return {
        "instruction": f"Beri rekomendasi makanan kategori {category}.",
        "answer": (
            f"Coba {merchant_name} di {merchant_area}."
            f" Menu andalan: {products_str}."
        )
    }

# Check dataset columns first
try:
    print(f"Loading dataset from {DATA_PATH}...")
    # csv loading with datasets library
    raw_ds = load_dataset("csv", data_files=DATA_PATH)["train"]
    print("Dataset columns:", raw_ds.column_names)
    
    # Shuffle and select a small subset for quick experimentation (500 samples)
    raw_ds = raw_ds.shuffle(seed=42).select(range(500)) 
    
    formatted_ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names)
    print("Dataset loaded and formatted successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating dummy dataset for demonstration purposes.")
    data = [
        {"product": "Nasi Goreng", "price": 15000, "category": "Indonesian", "merchant_name": "Warung A", "merchant_area": "Jakarta"},
        {"product": "Ayam Bakar", "price": 20000, "category": "Indonesian", "merchant_name": "Warung B", "merchant_area": "Bandung"},
        {"product": "Burger", "price": 35000, "category": "Western", "merchant_name": "Burger Spot", "merchant_area": "Surabaya"}
    ] * 50 # 150 samples
    raw_ds = Dataset.from_list(data)
    formatted_ds = raw_ds.map(format_example)
"""


code_tokenize = """# 4. Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    joined = [f"### Instruksi:\\n{ins}\\n\\n### Jawaban:\\n{ans}\\n"
              for ins, ans in zip(batch["instruction"], batch["answer"])]
    # Important update: Add padding="max_length" or padding=True with truncation
    # This ensures all items in the batch are the same length, preventing ValueError during training
    tokens = tokenizer(
        joined, 
        truncation=True, 
        max_length=512, 
        padding="max_length" 
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_ds = formatted_ds.map(tokenize, batched=True, remove_columns=["instruction", "answer"])
print("Tokenization complete.")
"""


code_model = """# 5. Load Model & Attach LoRA
# Using float16 if cuda is available to save memory, otherwise float32.
# avoiding bitsandbytes to ensure Windows compatibility unless explicitly installed.
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model {MODEL_ID} with dtype {torch_dtype}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map=device, 
    torch_dtype=torch_dtype
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"""

code_train = """# 6. Train
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, # Small batch size for memory safety
    gradient_accumulation_steps=8,
    num_train_epochs=1,          # Reduced to 1 epoch for quick experiment
    max_steps=100,               # Limit steps for demonstration (remove for full training)
    learning_rate=2e-4,
    fp16=(device == "cuda"),     # Enable fp16 only if cuda
    logging_steps=10,
    save_strategy="no",          # Skip intermediate saves for speed
    report_to="none"
)

print(f"Starting training on {device}...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=collator
)

trainer.train()
"""


code_save = """# 7. Save adapters and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA adapters saved to", OUTPUT_DIR)
"""

code_inference = """# 8. Inference
from peft import PeftModel

print("Loading model for inference...")
# Reload base model to ensure clean state (optional, but good practice)
# In a real Interactive session, you can just use 'model' from training, 
# but here we demonstrate loading the saved adapter.
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=device,
    torch_dtype=torch_dtype
)
model_to_test = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

# Test input - change category to test different things
test_category = "Indonesian"
test_instruction = f"Beri rekomendasi makanan kategori {test_category}."
prompt = f"### Instruksi:\\n{test_instruction}\\n\\n### Jawaban:\\n"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"Generating recommendation for {test_category}...")
outputs = model_to_test.generate(**inputs, max_new_tokens=100)
print("Result:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

# Construct the notebook cells
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# 1. Install Dependencies"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_install.splitlines(keepends=True)},
    
    {"cell_type": "markdown", "metadata": {}, "source": ["# 2. Imports and Setup"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_imports.splitlines(keepends=True)},
    
    {"cell_type": "markdown", "metadata": {}, "source": ["# 3. Load Data"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_dataset.splitlines(keepends=True)},
    
    {"cell_type": "markdown", "metadata": {}, "source": ["# 4. Tokenize"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_tokenize.splitlines(keepends=True)},
    
    {"cell_type": "markdown", "metadata": {}, "source": ["# 5. Model Prep (LoRA)"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_model.splitlines(keepends=True)},
    
    {"cell_type": "markdown", "metadata": {}, "source": ["# 6. Training"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_train.splitlines(keepends=True)},
    
    {"cell_type": "markdown", "metadata": {}, "source": ["# 7. Save"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_save.splitlines(keepends=True)},

    {"cell_type": "markdown", "metadata": {}, "source": ["# 8. Inference"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_inference.splitlines(keepends=True)},
]


notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Successfully updated {notebook_path}")
