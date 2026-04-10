#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge all learned Textual Inversion embeddings into a single file
for use in LoRA fine-tuning (e.g., derma_embeds.pt)
"""

import os
import glob
import torch
from safetensors.torch import load_file

TI_OUTPUT_DIR = "/home/edogan/Downloads/ertugrul/myenv/ti_lora_image_generation/outputs"
OUTPUT_PATH = os.path.join(TI_OUTPUT_DIR, "merged_embeds.pt")

pattern = os.path.join(TI_OUTPUT_DIR, "ti_*", "learned_embeds.safetensors")

merged_dict = {}
files = sorted(glob.glob(pattern))

if not files:
    raise FileNotFoundError(f"No embedding files found in pattern: {pattern}")

print(f"[INFO] Found {len(files)} embedding files.")
for file_path in files:
    print(f"[MERGE] Loading: {file_path}")

    data = load_file(file_path)

    for token, tensor in data.items():

        if tensor.dim() == 2 and tensor.shape[0] > 1:
            for i in range(tensor.shape[0]):
                t_name = token if i == 0 else f"{token}_{i}"
                if t_name in merged_dict:
                    print(f"[WARN] Token {t_name} already exists, overwriting.")
                merged_dict[t_name] = tensor[i]
        else:

            if tensor.dim() == 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
                
            if token in merged_dict:
                print(f"[WARN] Token {token} already exists, overwriting.")
            merged_dict[token] = tensor

os.makedirs(TI_OUTPUT_DIR, exist_ok=True)
torch.save(merged_dict, OUTPUT_PATH)

print(f"\n Aggregated embeddings saved at: {OUTPUT_PATH}")
print(f"Merged tokens: {list(merged_dict.keys())}")