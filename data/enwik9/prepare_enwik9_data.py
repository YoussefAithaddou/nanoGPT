import os
import pickle
import requests
import numpy as np
import re
import xml.etree.ElementTree as ET

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Set the file path for the enwik9 dataset
input_file_path = os.path.join(script_dir, 'enwik9')

# Download enwik9 dataset if not already present
if not os.path.exists(input_file_path):
    data_url = 'http://mattmahoney.net/dc/enwik9.zip'
    zip_path = os.path.join(script_dir, 'enwik9.zip')
    
    # Download and unzip in the current directory
    os.system(f"wget {data_url} -O {zip_path}")
    os.system(f"unzip {zip_path} -d {script_dir}")

# Read the dataset
with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    raw_data = f.read()

# Remove XML and non-text content
clean_data = re.sub(r'<.*?>', '', raw_data)  # Remove XML tags
clean_data = re.sub(r'[^\x20-\x7E]', ' ', clean_data)  # Keep printable ASCII characters
clean_data = re.sub(r'\s+', ' ', clean_data).strip()  # Normalize whitespace

# Print dataset size
print(f"Length of cleaned dataset in characters: {len(clean_data):,}")

# Get all unique characters
chars = sorted(list(set(clean_data)))
if '\n' not in chars:
    chars.append('\n')  # Adding newlines are in the vocabulary
chars = sorted(chars)



vocab_size = len(chars)
print(f"Unique characters: {''.join(chars)}")
print(f"Vocab size: {vocab_size:,}")

# Create character-to-index mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]  # Convert string to list of integer tokens

def decode(l):
    return ''.join([itos[i] for i in l])  # Convert list of integer tokens to string

# Split into train (90%), dev (5%), test (5%)
n = len(clean_data)
train_data = clean_data[:int(n * 0.90)]
val_data = clean_data[int(n * 0.90):int(n * 0.95)]
test_data = clean_data[int(n * 0.95):]

# Encode text as integers
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
test_ids = np.array(encode(test_data), dtype=np.uint16)

print(f"Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens | Test: {len(test_ids):,} tokens")

# Save to .bin files
train_ids.tofile(os.path.join(script_dir, 'train.bin'))
val_ids.tofile(os.path.join(script_dir, 'val.bin'))
test_ids.tofile(os.path.join(script_dir, 'test.bin'))

# Save the meta info for encoding/decoding
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(script_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete! Files saved: train.bin, val.bin, test.bin, meta.pkl")
