#!/usr/bin/env python3
import json
import os
from pathlib import Path

# Define paths
data_dir = "/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data"
cold_start_dir = os.path.join(data_dir, "cold_start_9k")
phase_dir = os.path.join(data_dir, "phase_cold_start_9k")

# Create output directory
os.makedirs(phase_dir, exist_ok=True)

# Load datasets
print("Loading datasets...")
with open(os.path.join(cold_start_dir, "train_type1.json"), "r") as f:
    type1_data = json.load(f)

with open(os.path.join(cold_start_dir, "train_type2.json"), "r") as f:
    type2_data = json.load(f)

with open(os.path.join(cold_start_dir, "train_type3.json"), "r") as f:
    type3_data = json.load(f)

print(f"Loaded: type1={len(type1_data)}, type2={len(type2_data)}, type3={len(type3_data)}")

# Calculate split points
type3_third = len(type3_data) // 3
type2_half = len(type2_data) // 2

# Dataset 1: 1/3 of type3
dataset1 = type3_data[:type3_third]

# Dataset 2: another 1/3 of type3 + 1/2 of type2
dataset2 = type3_data[type3_third:2*type3_third] + type2_data[:type2_half]

# Dataset 3: remaining 1/3 of type3 + remaining 1/2 of type2 + all of type1
dataset3 = type3_data[2*type3_third:] + type2_data[type2_half:] + type1_data

print(f"\nCreated datasets: dataset1={len(dataset1)}, dataset2={len(dataset2)}, dataset3={len(dataset3)}")

# Save datasets
output_files = {
    "phase1_train.json": dataset1,
    "phase2_train.json": dataset2,
    "phase3_train.json": dataset3,
}

for filename, data in output_files.items():
    filepath = os.path.join(phase_dir, filename)
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved: {filepath} ({len(data)} samples)")

print("\nDatasets created successfully!")
