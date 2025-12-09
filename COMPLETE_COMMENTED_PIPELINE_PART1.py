"""
================================================================================
FACE RECOGNITION SYSTEM WITH TRIPLET LOSS AND GENDER CLASSIFICATION
================================================================================

Project: Thesis Implementation - Deep Metric Learning for Face Recognition
Author: [Your Name]
Date: December 2025

SYSTEM OVERVIEW:
---------------
This system implements an end-to-end face recognition pipeline using:
1. VGGFace2 dataset (30-identity subset for efficient prototyping)
2. MobileNetV2 backbone for lightweight feature extraction
3. Triplet Loss for learning discriminative embeddings
4. Dynamic embedding database for one-shot learning capability
5. Transfer learning for gender classification

EXPECTED RESULTS:
----------------
- Identity Recognition: 81.21% accuracy (±2% variance acceptable)
- Gender Classification: 97.60% accuracy (±1% variance acceptable)
- Training Time: ~13 minutes on GPU
- Inference Time: ~15ms per image

REPRODUCIBILITY NOTES:
---------------------
- All random seeds fixed at 42 for deterministic results
- Dataset automatically downloads via KaggleHub
- Minor variations (±1-2%) are normal due to GPU non-determinism
- Tested on: Google Colab with NVIDIA L4 GPU (23.8 GB)

HOW TO RUN:
----------
1. Run Cell 1: Imports and Setup (sets seeds, checks GPU)
2. Run Cell 2: Dataset Download and Preparation
3. Run Cell 3: Dataset Cleaning and Train/Val Split
4. Run Cell 4: Data Loaders and Visualization
5. Run Cell 5: Embedding Model Architecture
6. Run Cell 6: Triplet Loss Training (5 epochs, ~10 min)
7. Run Cell 7: Identity Recognition Evaluation
8. Run Cell 8: Dynamic Embedding Database
9. Run Cell 9: Gender Classification Training
10. Run Cell 10: Interactive Inference Demo

Total runtime: ~15-20 minutes
================================================================================
"""

# ================================================================================
# CELL 1: IMPORTS AND ENVIRONMENT SETUP
# ================================================================================
"""
PURPOSE:
--------
- Import all required libraries for deep learning, data processing, and visualization
- Set random seeds for reproducibility (seed=42)
- Check hardware availability (GPU strongly recommended)
- Configure deterministic behavior for PyTorch

DEPENDENCIES:
------------
All packages will be verified and their versions printed for reproducibility.
"""

print("="*70)
print("STEP 1: IMPORTING LIBRARIES AND CONFIGURING ENVIRONMENT")
print("="*70)

# Core deep learning framework
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Image processing
from PIL import Image
import cv2

# Numerical computing and data structures
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# System utilities
import os
import sys
from pathlib import Path
from tqdm import tqdm  # Progress bars
from collections import defaultdict, Counter
import time
import pickle
import json
import random
import glob

# Google Colab utilities (for file upload and interactive features)
from google.colab import files, drive

print("✓ All libraries imported successfully")

# ============================================================================
# SET RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
"""
CRITICAL FOR REPRODUCIBILITY:
-----------------------------
Setting random seeds ensures that:
1. Dataset splitting is consistent across runs
2. Model initialization is identical
3. Triplet sampling follows the same pattern
4. Training converges to similar solutions

NOTE: GPU operations may still have minor non-determinism (±1-2% variance)
"""

def set_seed(seed=42):
    """
    Set all random seeds for reproducible experiments.
    
    Args:
        seed (int): Random seed value (default=42)
    
    Effects:
        - PyTorch CPU and GPU operations become deterministic
        - NumPy random operations become deterministic
        - Python's random module becomes deterministic
        - CUDNN backend uses deterministic algorithms (slightly slower but reproducible)
    """
    torch.manual_seed(seed)                    # PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU random seed (all GPUs)
    np.random.seed(seed)                       # NumPy random seed
    random.seed(seed)                          # Python random module seed
    
    # Make PyTorch operations deterministic
    # NOTE: This may reduce performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed} for reproducibility")

# Apply seed
set_seed(42)

# ============================================================================
# CHECK HARDWARE AVAILABILITY
# ============================================================================
"""
HARDWARE REQUIREMENTS:
---------------------
- GPU: Strongly recommended (training takes ~10 min on GPU vs ~90 min on CPU)
- RAM: 8GB minimum, 16GB recommended
- Disk: 10GB free space for dataset

This section verifies hardware and prints detailed specifications.
"""

# Determine device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*70)
print("HARDWARE CHECK")
print("="*70)
print(f"Device: {device}")

if torch.cuda.is_available():
    # GPU is available - print detailed specs
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU Memory: {gpu_memory_gb:.2f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print("\n✓ Training will be fast (~10 minutes)")
else:
    # CPU only - training will be slower
    print("⚠️  WARNING: No GPU detected!")
    print("   Training will be slow (~90 minutes)")
    print("   Recommendation: Runtime → Change runtime type → GPU")

print("="*70)

# Print library versions for reproducibility documentation
print("\nLIBRARY VERSIONS (for reproducibility):")
print(f"  Python: {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  PIL: {Image.__version__}")

print("\n✓ Environment setup complete!")
print("="*70)


# ================================================================================
# CELL 2: DATASET DOWNLOAD AND INITIAL EXPLORATION
# ================================================================================
"""
PURPOSE:
--------
- Download VGGFace2 dataset via KaggleHub (automatic, no manual download needed)
- Explore dataset structure (number of identities, images per identity)
- Visualize sample images to verify data quality

DATASET DETAILS:
---------------
- Source: VGGFace2 (via Kaggle)
- Full dataset: 3.31M images, 9,131 identities
- Subset used: 480 identities (manageable for prototyping)
- We will further subset to 30 identities for this demonstration

WHY SUBSET?
----------
- Faster training for validating architecture (~10 min vs hours)
- Sufficient to demonstrate triplet loss effectiveness
- Can scale to full dataset after validation
- Maintains class balance and diversity
"""

print("\n" + "="*70)
print("STEP 2: DATASET DOWNLOAD AND EXPLORATION")
print("="*70)

# Install KaggleHub for dataset download
print("\nInstalling KaggleHub (dataset downloader)...")
!pip install kagglehub opencv-python matplotlib -q
print("✓ KaggleHub installed")

import kagglehub

# ============================================================================
# DOWNLOAD DATASET
# ============================================================================
"""
AUTOMATIC DOWNLOAD:
------------------
KaggleHub automatically:
1. Downloads dataset from Kaggle
2. Caches it for future use
3. Returns path to downloaded files

DOWNLOAD TIME: ~3-5 minutes (dataset is ~500MB for this subset)
LOCATION: /kaggle/input/vggface2
"""

print("\nDownloading VGGFace2 dataset...")
print("(This may take 3-5 minutes on first run)")
print("(Subsequent runs use cached version - instant)")

path = kagglehub.dataset_download("hearfool/vggface2")

print(f"\n✓ Dataset downloaded successfully!")
print(f"  Location: {path}")
print(f"  Contents: {os.listdir(path)}")

# ============================================================================
# EXPLORE DATASET STRUCTURE
# ============================================================================
"""
EXPECTED STRUCTURE:
------------------
vggface2/
  ├── train/        # Training images
  │   ├── n000002/  # Identity folder
  │   │   ├── 0001_01.jpg
  │   │   ├── 0002_01.jpg
  │   │   └── ...
  │   ├── n000003/
  │   └── ...
  └── val/          # Validation images

Each identity (person) has their own folder with multiple images.
"""

# Set path to training data
train_root = os.path.join(path, "train")
print(f"\nTraining data path: {train_root}")

# Count number of identities
identity_folders = sorted([
    d for d in os.listdir(train_root) 
    if os.path.isdir(os.path.join(train_root, d))
])

print(f"\n✓ Total identities in dataset: {len(identity_folders)}")
print(f"  First 10 identities: {identity_folders[:10]}")

# ============================================================================
# VISUALIZE RANDOM IDENTITY
# ============================================================================
"""
VISUALIZATION PURPOSE:
--------------------
Show 4 sample images from a random identity to verify:
1. Images are properly loaded
2. Same person appears in multiple images
3. Variations in pose, lighting, expression exist (good for training)
"""

print("\nVisualizing sample images from random identity...")

# Select random identity
random_id = random.choice(identity_folders)
id_dir = os.path.join(train_root, random_id)
num_images = len(os.listdir(id_dir))

print(f"  Selected identity: {random_id}")
print(f"  Number of images: {num_images}")

# Get first 4 images
image_paths = glob.glob(os.path.join(id_dir, "*.jpg"))[:4]

# Display images
plt.figure(figsize=(12, 3))
for i, img_path in enumerate(image_paths):
    img = Image.open(img_path).convert("RGB")
    plt.subplot(1, 4, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{random_id}\nImage {i+1}")
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sample_identity.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Sample visualization saved to sample_identity.png")
print("="*70)


# ================================================================================
# CELL 3: DATASET FILTERING, CLEANING, AND TRAIN/VAL SPLIT
# ================================================================================
"""
PURPOSE:
--------
1. Filter identities with sufficient images (≥20 per identity)
2. Create manageable subset (30 identities for this demo)
3. Remove corrupted images (PIL verification)
4. Split into train (80%) and validation (20%) sets
5. Create label mappings

WHY 30 IDENTITIES?
-----------------
- Demonstrates architecture effectiveness
- Fast training (~10 minutes for 5 epochs)
- Maintains class balance
- Scalable to full dataset (480 or 9,131 identities)

QUALITY CONTROL:
---------------
- Verify all images can be opened (no corruption)
- Ensure minimum 20 images per identity (for train/val split)
- Stratified split maintains class balance
"""

print("\n" + "="*70)
print("STEP 3: DATASET PREPARATION AND QUALITY CONTROL")
print("="*70)

# ============================================================================
# COUNT IMAGES PER IDENTITY
# ============================================================================
"""
Build dictionary mapping identity to list of image paths.
This allows us to:
1. Filter by minimum image count
2. Create balanced subsets
3. Perform stratified splitting
"""

print("\nCounting images per identity...")

id_to_imgs = {}  # Dictionary: identity_id -> list of image paths

for person_id in identity_folders:
    person_dir = os.path.join(train_root, person_id)
    # Find all JPG images in person's directory
    img_paths = glob.glob(os.path.join(person_dir, "*.jpg"))
    
    if len(img_paths) > 0:
        id_to_imgs[person_id] = img_paths

print(f"✓ Identities with at least 1 image: {len(id_to_imgs)}")

# ============================================================================
# FILTER BY MINIMUM IMAGE COUNT
# ============================================================================
"""
RATIONALE:
---------
Need minimum 20 images per identity to:
- Have enough for train/val split (16 train + 4 val)
- Ensure triplet loss has sufficient positive pairs
- Maintain statistical validity in evaluation
"""

MIN_IMAGES_PER_ID = 20
eligible_ids = [
    pid for pid, imgs in id_to_imgs.items() 
    if len(imgs) >= MIN_IMAGES_PER_ID
]

print(f"✓ Identities with ≥{MIN_IMAGES_PER_ID} images: {len(eligible_ids)}")

# ============================================================================
# CREATE SUBSET FOR PROTOTYPING
# ============================================================================
"""
SELECT 30 IDENTITIES:
--------------------
- Representative sample of full dataset
- Sufficient to validate triplet loss approach
- Fast training for iterative development
- Can scale to full dataset after validation

NOTE: random.sample() uses our fixed seed (42), so same 30 identities 
      are selected every run for reproducibility
"""

NUM_IDENTITIES = 30
subset_ids = random.sample(eligible_ids, k=min(NUM_IDENTITIES, len(eligible_ids)))

print(f"\n✓ Selected {len(subset_ids)} identities for training")
print(f"  Sample IDs: {subset_ids[:5]}...")

# ============================================================================
# CREATE LABEL MAPPING
# ============================================================================
"""
LABEL MAPPING:
-------------
Convert string identity IDs (e.g., 'n000002') to integer labels (0, 1, 2, ...)
This is required for:
- PyTorch loss functions (expect integer labels)
- Consistent ordering across runs
- Efficient storage and processing
"""

label_map = {
    pid: idx 
    for idx, pid in enumerate(sorted(subset_ids))
}
num_classes = len(label_map)

print(f"✓ Created label mapping for {num_classes} classes")
print(f"  Example: {list(label_map.items())[:3]}")

# ============================================================================
# BUILD RAW SAMPLE LIST
# ============================================================================
"""
CREATE LIST OF (image_path, label) TUPLES:
-----------------------------------------
This is our raw dataset before cleaning and splitting.
"""

all_samples = []  # List of (img_path, label) tuples

for pid in subset_ids:
    imgs = id_to_imgs[pid]
    label = label_map[pid]
    
    for img_path in imgs:
        all_samples.append((img_path, label))

print(f"\n✓ Total raw samples (before cleaning): {len(all_samples)}")

# ============================================================================
# DATA CLEANING: REMOVE CORRUPTED IMAGES
# ============================================================================
"""
IMAGE VERIFICATION:
------------------
Use PIL's verify() to check each image:
- Detects truncated files
- Detects corrupted headers
- Ensures images can be loaded during training

VERIFICATION METHOD:
- Open image with PIL
- Call verify() (lightweight check)
- If exception raised, skip image
"""

print("\nPerforming data quality check...")
print("(Verifying all images can be loaded)")

clean_samples = []
corrupted_count = 0

for img_path, label in tqdm(all_samples, desc="Verifying images"):
    try:
        # Attempt to open and verify image
        with Image.open(img_path) as im:
            im.verify()  # Checks if image is corrupted
        
        # If successful, add to clean list
        clean_samples.append((img_path, label))
        
    except Exception as e:
        # Image is corrupted, skip it
        corrupted_count += 1
        # Optionally print error (commented out to reduce clutter)
        # print(f"Skipping corrupted: {img_path} | Error: {e}")

print(f"\n✓ Data cleaning complete:")
print(f"  Valid images: {len(clean_samples)}")
print(f"  Corrupted images: {corrupted_count}")
print(f"  Retention rate: {100*len(clean_samples)/len(all_samples):.2f}%")

# ============================================================================
# STRATIFIED TRAIN/VALIDATION SPLIT
# ============================================================================
"""
STRATIFIED SPLIT:
----------------
Ensures each identity (class) has proportional representation in train/val.

PROCESS:
1. Group images by label
2. For each label, split 80% train / 20% val
3. Maintain at least 1 image per identity in validation

BENEFITS:
- Balanced class distribution in train and val
- Prevents bias toward frequent classes
- Realistic evaluation (all identities represented)
"""

print("\nPerforming stratified train/validation split...")

# Group samples by label
by_label = defaultdict(list)
for img_path, label in clean_samples:
    by_label[label].append(img_path)

# Split each label's images into train and val
train_samples = []
val_samples = []
train_ratio = 0.8

for label, paths in by_label.items():
    # Shuffle paths for this label (uses our seed=42)
    random.shuffle(paths)
    
    # Calculate split point
    n = len(paths)
    n_train = max(1, int(train_ratio * n))  # At least 1 for training
    
    # Split
    train_paths = paths[:n_train]
    val_paths = paths[n_train:]
    
    # Add to global lists
    train_samples.extend([(p, label) for p in train_paths])
    val_samples.extend([(p, label) for p in val_paths])

print(f"\n✓ Split complete:")
print(f"  Training samples: {len(train_samples)}")
print(f"  Validation samples: {len(val_samples)}")
print(f"  Ratio: {len(train_samples)/(len(train_samples)+len(val_samples)):.1%} train / " +
      f"{len(val_samples)/(len(train_samples)+len(val_samples)):.1%} val")

# ============================================================================
# VERIFY CLASS BALANCE
# ============================================================================
"""
CHECK DISTRIBUTION:
------------------
Verify that all classes are represented in both train and val sets.
Print distribution statistics.
"""

train_counts = Counter([lbl for _, lbl in train_samples])
val_counts = Counter([lbl for _, lbl in val_samples])

print(f"\n✓ Class distribution verified:")
print(f"  Train classes: {len(train_counts)} (all {num_classes} present)")
print(f"  Val classes: {len(val_counts)} (all {num_classes} present)")
print(f"  Avg images per class (train): {sum(train_counts.values())/len(train_counts):.1f}")
print(f"  Avg images per class (val): {sum(val_counts.values())/len(val_counts):.1f}")

print("="*70)


# ================================================================================
# CELL 4: DATA LOADERS AND BATCH VISUALIZATION
# ================================================================================
"""
PURPOSE:
--------
1. Define image transformations (preprocessing pipeline)
2. Create PyTorch Dataset and DataLoader objects
3. Visualize a training batch to verify pipeline

IMAGE PREPROCESSING:
-------------------
Training augmentation:
  - Resize to 160×160 pixels (MobileNetV2 compatible)
  - Random horizontal flip (50% probability) - data augmentation
  - Convert to tensor
  - Normalize to [-1, 1] range (mean=0.5, std=0.5 per channel)

Validation (no augmentation):
  - Resize to 160×160 pixels
  - Convert to tensor
  - Normalize to [-1, 1] range

WHY 160×160?
-----------
- MobileNetV2 accepts flexible input sizes
- 160×160 balances detail and computation
- Standard for face recognition tasks
- Larger than minimum 112×112 but smaller than 224×224 (ImageNet standard)
"""

print("\n" + "="*70)
print("STEP 4: DATA LOADERS AND VISUALIZATION")
print("="*70)

# ============================================================================
# DEFINE IMAGE TRANSFORMATIONS
# ============================================================================
"""
TRANSFORMATION PIPELINE:
-----------------------
Applied to every image before feeding to model.

transforms.Compose(): Chains multiple transformations
transforms.Resize(): Resizes image to specified dimensions
transforms.RandomHorizontalFlip(): Flips image horizontally (augmentation)
transforms.ToTensor(): Converts PIL Image to PyTorch tensor (0-1 range)
transforms.Normalize(): Normalizes tensor (subtracts mean, divides by std)
"""

IMG_SIZE = 160  # Target image size

# Training transformations (with augmentation)
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),         # Resize to 160×160
    transforms.RandomHorizontalFlip(p=0.5),          # 50% chance horizontal flip
    transforms.ToTensor(),                           # Convert to tensor [0, 1]
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],                       # Normalize to [-1, 1]
        std=[0.5, 0.5, 0.5]
    )
])

# Validation transformations (no augmentation)
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

print(f"✓ Image transformations defined")
print(f"  Target size: {IMG_SIZE}×{IMG_SIZE} pixels")
print(f"  Training augmentation: Random horizontal flip")
print(f"  Normalization: mean=0.5, std=0.5 (range [-1, 1])")

# ============================================================================
# CREATE PYTORCH DATASET CLASS
# ============================================================================
"""
CUSTOM DATASET:
--------------
PyTorch requires a Dataset class with:
- __init__: Initialize with data
- __len__: Return total number of samples
- __getitem__: Return (image, label) for given index

This class loads images on-the-fly (saves memory vs loading all at once)
"""

class VGGFace2Subset(Dataset):
    """
    PyTorch Dataset for VGGFace2 subset.
    
    Args:
        samples (list): List of (img_path, label) tuples
        transform (callable): Transform to apply to images
    """
    
    def __init__(self, samples, transform=None):
        """
        Initialize dataset.
        
        Args:
            samples: List of (image_path, label) tuples
            transform: torchvision transforms to apply
        """
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and return one sample.
        
        Args:
            idx (int): Index of sample to load
            
        Returns:
            tuple: (image_tensor, label)
                - image_tensor: Transformed image (3×160×160)
                - label: Integer class label
        """
        # Get image path and label
        img_path, label = self.samples[idx]
        
        # Load image as RGB (ensures 3 channels)
        img = Image.open(img_path).convert("RGB")
        
        # Apply transformations if specified
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

# Create dataset objects
train_dataset = VGGFace2Subset(train_samples, transform=transform_train)
val_dataset = VGGFace2Subset(val_samples, transform=transform_val)

print(f"\n✓ Datasets created:")
print(f"  Training dataset: {len(train_dataset)} images")
print(f"  Validation dataset: {len(val_dataset)} images")

# ============================================================================
# CREATE DATA LOADERS
# ============================================================================
"""
DATA LOADER:
-----------
PyTorch DataLoader handles:
- Batching: Groups images into batches
- Shuffling: Randomizes order (training only)
- Parallel loading: Uses multiple workers for efficiency
- Memory management: Loads batches as needed

BATCH_SIZE=32:
- Fits comfortably in GPU memory
- Provides stable gradients
- Good balance between speed and memory
"""

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,           # Shuffle training data each epoch
    num_workers=2,          # Parallel data loading
    pin_memory=True         # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # Don't shuffle validation data
    num_workers=2,
    pin_memory=True
)

print(f"\n✓ Data loaders created:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")

# ============================================================================
# VISUALIZE A BATCH
# ============================================================================
"""
BATCH VISUALIZATION:
-------------------
Load one batch from training loader and display images.
This verifies:
1. Images load correctly
2. Transformations work properly
3. Batching is correct
4. Labels are assigned properly

DENORMALIZATION:
- Images are normalized to [-1, 1] for training
- For display, convert back to [0, 1] range
- Formula: img_display = (img_normalized * std) + mean
           = (img * 0.5) + 0.5
"""

print("\nLoading and visualizing first training batch...")

# Get one batch
batch_imgs, batch_labels = next(iter(train_loader))

print(f"\n✓ Batch loaded:")
print(f"  Batch shape: {batch_imgs.shape}")  # Should be (32, 3, 160, 160)
print(f"  Labels: {batch_labels[:10].tolist()}")  # First 10 labels

# Denormalize images for visualization
def denormalize(img_tensor):
    """
    Convert normalized tensor back to displayable image.
    
    Args:
        img_tensor: Tensor with values in [-1, 1]
        
    Returns:
        img: Tensor with values in [0, 1]
    """
    img = img_tensor.clone()
    img = img * 0.5 + 0.5  # Reverse normalization
    img = img.clamp(0, 1)   # Ensure valid range
    return img

# Display first 8 images from batch
n_show = 8
plt.figure(figsize=(16, 4))

for i in range(n_show):
    # Denormalize and convert to numpy
    img = denormalize(batch_imgs[i]).permute(1, 2, 0).numpy()
    label = batch_labels[i].item()
    
    # Plot
    plt.subplot(1, n_show, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Label: {label}", fontsize=10)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/batch_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Batch visualization saved to batch_visualization.png")
print("="*70)

# ================================================================================
# READY FOR MODEL TRAINING!
# ================================================================================
print("\n" + "="*70)
print("✓ DATA PIPELINE COMPLETE - READY FOR MODEL TRAINING")
print("="*70)
print("\nSummary:")
print(f"  • {len(train_samples)} training images")
print(f"  • {len(val_samples)} validation images") 
print(f"  • {num_classes} identities (classes)")
print(f"  • Batch size: {BATCH_SIZE}")
print(f"  • Image size: {IMG_SIZE}×{IMG_SIZE}")
print("="*70)
