import os
import pickle
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Load the meta file
meta_path = os.path.join(script_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

def decode(l):
    return ''.join([itos[i] for i in l])  # Convert list of integer tokens to string

# Load encoded dataset
train_path = os.path.join(script_dir, 'train.bin')
val_path = os.path.join(script_dir, 'val.bin')
test_path = os.path.join(script_dir, 'test.bin')

train_ids = np.fromfile(train_path, dtype=np.uint16)
val_ids = np.fromfile(val_path, dtype=np.uint16)
test_ids = np.fromfile(test_path, dtype=np.uint16)

# Show samples
print(f"Train dataset size: {len(train_ids):,} tokens")
print(f"Validation dataset size: {len(val_ids):,} tokens")
print(f"Test dataset size: {len(test_ids):,} tokens")

# Show a sample from the train set
sample_size = 100
sample_tokens = train_ids[:sample_size]
decoded_text = decode(sample_tokens)

print(f"\nFirst {sample_size} tokens from train data:\n{sample_tokens}")
print(f"\nDecoded text:\n{decoded_text}")
