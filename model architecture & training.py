"""
================================================================================
PART 2: MODEL ARCHITECTURE AND TRAINING
================================================================================

This section contains:
- Cell 5: MobileNetV2 Embedding Network Architecture
- Cell 6: Triplet Loss Training (5 epochs)
- Cell 7: Model Evaluation and Gallery-Probe Testing
"""

# ================================================================================
# CELL 5: EMBEDDING NETWORK ARCHITECTURE
# ================================================================================
"""
PURPOSE:
--------
Define the face embedding network architecture using MobileNetV2 backbone.

ARCHITECTURE DESIGN:
-------------------
Input (160×160×3) → MobileNetV2 Features → Global Avg Pool → 
Linear (1280→128) → L2 Normalize → 128-dim Embedding

WHY MOBILENETV2?
---------------
1. Lightweight: Only 3.5M parameters (vs ResNet50's 25M)
2. Fast inference: ~15ms per image on GPU
3. Mobile-deployable: ~4MB model size
4. Pre-trained: ImageNet weights provide good initialization
5. Proven: Widely used in production face recognition

WHY 128-DIM EMBEDDINGS?
-----------------------
1. Balance between expressiveness and efficiency
2. Standard in face recognition (FaceNet uses 128 or 512)
3. Sufficient for 30-class problem (can scale to thousands)
4. Faster similarity computations than 512-dim
5. Less prone to overfitting than higher dimensions

WHY L2 NORMALIZATION?
--------------------
1. Maps embeddings to unit hypersphere (all have length 1)
2. Enables cosine similarity as distance metric
3. Simplifies threshold selection
4. Standard practice in metric learning
5. Makes triplet loss training more stable
"""

print("\n" + "="*70)
print("STEP 5: EMBEDDING NETWORK ARCHITECTURE")
print("="*70)

class MobileNetEmbeddingNet(nn.Module):
    """
    Face embedding network using MobileNetV2 backbone.
    
    Architecture:
    ------------
    MobileNetV2 (pretrained) → GlobalAvgPool → Linear → L2-Normalize
    
    Input:  (B, 3, 160, 160) - Batch of RGB images
    Output: (B, 128) - L2-normalized embeddings
    
    Args:
        embedding_size (int): Dimension of output embeddings (default=128)
    """
    
    def __init__(self, embedding_size=128):
        """
        Initialize embedding network.
        
        Steps:
        1. Load pretrained MobileNetV2 from torchvision
        2. Extract feature extractor (convolutional layers)
        3. Add global average pooling
        4. Add projection head (1280 → embedding_size)
        5. L2 normalization will be applied in forward()
        """
        super(MobileNetEmbeddingNet, self).__init__()
        
        # ================================================================
        # BACKBONE: MOBILENETV2
        # ================================================================
        """
        Load pretrained MobileNetV2 from torchvision.
        
        MobileNetV2 architecture:
        - Depthwise separable convolutions (efficient)
        - Inverted residual blocks
        - Linear bottlenecks
        - Final output: 1280 channels
        
        Pretrained weights from ImageNet provide:
        - Good initialization for face features
        - Faster convergence
        - Better generalization
        """
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1  # Official pretrained weights
        )
        
        # Extract feature extractor (all conv layers)
        # This is everything except the final classifier
        self.features = backbone.features
        
        print("✓ MobileNetV2 backbone loaded (pretrained on ImageNet)")
        print(f"  Output channels: 1280")
        
        # ================================================================
        # GLOBAL AVERAGE POOLING
        # ================================================================
        """
        Reduces spatial dimensions to 1×1 while keeping channels.
        
        Input:  (B, 1280, H', W') where H', W' depend on input size
        Output: (B, 1280, 1, 1)
        
        Benefits:
        - Spatial invariance (face can be anywhere in image)
        - Reduces parameters in next layer
        - Standard practice in CNN architectures
        """
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ================================================================
        # EMBEDDING PROJECTION HEAD
        # ================================================================
        """
        Projects 1280-dim features to embedding_size dimensions.
        
        This is a fully connected layer that learns the optimal
        projection for face recognition task.
        
        Input:  (B, 1280)
        Output: (B, embedding_size)
        """
        self.embedding_layer = nn.Linear(1280, embedding_size)
        
        print(f"✓ Projection head: 1280 → {embedding_size} dimensions")
        
        # ================================================================
        # L2 NORMALIZATION (applied in forward())
        # ================================================================
        """
        L2 normalization will be applied to embeddings in forward().
        
        This ensures all embeddings lie on the unit hypersphere:
        ||embedding|| = 1 for all embeddings
        
        Benefits:
        - Cosine similarity = dot product (simplified computation)
        - Consistent scale for threshold selection
        - Prevents magnitude dominance in distance calculations
        """
        print(f"✓ L2 normalization enabled (unit hypersphere embeddings)")
    
    def forward(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input images (B, 3, 160, 160)
            
        Returns:
            embeddings: L2-normalized embeddings (B, embedding_size)
        
        Process:
        1. Extract features via MobileNetV2: (B,3,160,160) → (B,1280,5,5)
        2. Global average pooling: (B,1280,5,5) → (B,1280,1,1)
        3. Flatten: (B,1280,1,1) → (B,1280)
        4. Project to embedding space: (B,1280) → (B,128)
        5. L2 normalize: embedding → embedding / ||embedding||
        """
        # Step 1: Convolutional feature extraction
        x = self.features(x)  # (B, 1280, H', W')
        
        # Step 2: Global average pooling
        x = self.avgpool(x)   # (B, 1280, 1, 1)
        
        # Step 3: Flatten to vector
        x = x.view(x.size(0), -1)  # (B, 1280)
        
        # Step 4: Project to embedding space
        x = self.embedding_layer(x)  # (B, 128)
        
        # Step 5: L2 normalize (critical for metric learning!)
        # This ensures all embeddings have unit length
        x = F.normalize(x, p=2, dim=1)  # (B, 128), ||x|| = 1
        
        return x

# ============================================================================
# INSTANTIATE AND VERIFY MODEL
# ============================================================================
"""
Create model instance and verify it works correctly.
"""

print("\nInitializing model...")

# Create model
model = MobileNetEmbeddingNet(embedding_size=128)

# Move to GPU if available
model = model.to(device)

print(f"✓ Model moved to {device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params*4/1e6:.1f} MB (float32)")

# ============================================================================
# TEST MODEL WITH DUMMY BATCH
# ============================================================================
"""
Verify model works correctly by passing dummy data through it.
"""

print("\nTesting model with dummy batch...")

# Create dummy input (batch of 8 images)
dummy_input = torch.randn(8, 3, 160, 160).to(device)

# Forward pass
with torch.no_grad():
    embeddings = model(dummy_input)

print(f"\n✓ Model test successful:")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {embeddings.shape}")
print(f"  Output dtype: {embeddings.dtype}")

# Verify L2 normalization
norms = torch.norm(embeddings, p=2, dim=1)
print(f"  L2 norms: {norms[:3].cpu().numpy()}")  # Should all be 1.0
print(f"  All norms ≈ 1.0: {torch.allclose(norms, torch.ones_like(norms))}")

print("="*70)


# ================================================================================
# CELL 6: TRIPLET DATASET AND LOSS TRAINING
# ================================================================================
"""
PURPOSE:
--------
1. Create Triplet Dataset for online triplet sampling
2. Define Triplet Loss function
3. Train model for 5 epochs
4. Save best checkpoint

TRIPLET LOSS EXPLAINED:
-----------------------
Traditional classification doesn't scale to new identities.
Triplet loss learns a metric space where:
- Same person → small distance
- Different people → large distance

For each sample (anchor):
- Positive: Different image of SAME person
- Negative: Image of DIFFERENT person

Loss = max(d(anchor,positive) - d(anchor,negative) + margin, 0)

WHY MARGIN?
----------
- Margin enforces minimum separation between classes
- Prevents trivial solution (all embeddings at single point)
- Typical values: 0.2-0.5 (we use 0.3)

ONLINE TRIPLET FORMATION:
-------------------------
Instead of pre-computing all triplets (which grows cubically),
we form triplets on-the-fly during training:
1. Sample anchor image
2. Sample positive from same identity
3. Sample negative from different identity
"""

print("\n" + "="*70)
print("STEP 6: TRIPLET LOSS TRAINING")
print("="*70)

# ============================================================================
# TRIPLET DATASET CLASS
# ============================================================================
"""
Creates triplets on-the-fly during training.
"""

class TripletVGGFace2(Dataset):
    """
    Dataset for triplet loss training.
    
    For each __getitem__ call, returns:
    - Anchor: Random image from dataset
    - Positive: Different image of SAME person as anchor
    - Negative: Image from DIFFERENT person
    
    Args:
        samples (list): List of (img_path, label) tuples
        transform (callable): Transformations to apply
    """
    
    def __init__(self, samples, transform=None):
        """
        Initialize triplet dataset.
        
        Steps:
        1. Store samples and transforms
        2. Build label_to_indices mapping for fast triplet sampling
        3. Extract unique labels
        """
        self.samples = samples
        self.transform = transform
        
        # Build index: label → list of sample indices
        # This allows O(1) lookup of all images for a given person
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # List of all unique labels
        self.labels = list(self.label_to_indices.keys())
        
        print(f"✓ Triplet dataset initialized")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Unique labels: {len(self.labels)}")
        print(f"  Avg samples per label: {len(self.samples)/len(self.labels):.1f}")
    
    def __len__(self):
        """Return total number of samples (same as base dataset)."""
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Return triplet: (anchor, positive, negative) images.
        
        Args:
            index (int): Index for anchor image
            
        Returns:
            tuple: (anchor_img, positive_img, negative_img)
                All images are transformed tensors (3×160×160)
        
        Process:
        1. Use index to get anchor image and its label
        2. Sample positive: different index, same label
        3. Sample negative: random image from different label
        4. Load and transform all three images
        """
        # ==============================================================
        # ANCHOR: Image at given index
        # ==============================================================
        anchor_path, anchor_label = self.samples[index]
        anchor_img = Image.open(anchor_path).convert("RGB")
        
        # ==============================================================
        # POSITIVE: Different image, SAME person
        # ==============================================================
        # Get all indices for this label
        positive_indices = self.label_to_indices[anchor_label]
        
        if len(positive_indices) == 1:
            # Only one image for this person, use same image
            # (This shouldn't happen with our filtering, but handle it)
            pos_index = positive_indices[0]
        else:
            # Sample different image from same person
            pos_index = random.choice([
                i for i in positive_indices if i != index
            ])
        
        positive_path, _ = self.samples[pos_index]
        positive_img = Image.open(positive_path).convert("RGB")
        
        # ==============================================================
        # NEGATIVE: Random image from DIFFERENT person
        # ==============================================================
        # Sample random label different from anchor
        neg_label = random.choice([
            lbl for lbl in self.labels if lbl != anchor_label
        ])
        
        # Sample random image from that label
        neg_index = random.choice(self.label_to_indices[neg_label])
        negative_path, _ = self.samples[neg_index]
        negative_img = Image.open(negative_path).convert("RGB")
        
        # ==============================================================
        # APPLY TRANSFORMATIONS
        # ==============================================================
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img

# Create triplet dataset
print("\nCreating triplet dataset...")
triplet_train_dataset = TripletVGGFace2(train_samples, transform=transform_train)

# Create triplet data loader
TRIPLET_BATCH_SIZE = 32

triplet_train_loader = DataLoader(
    triplet_train_dataset,
    batch_size=TRIPLET_BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

print(f"\n✓ Triplet data loader created:")
print(f"  Batch size: {TRIPLET_BATCH_SIZE}")
print(f"  Batches per epoch: {len(triplet_train_loader)}")

# Verify triplet loading
print("\nVerifying triplet sampling...")
a, p, n = next(iter(triplet_train_loader))
print(f"✓ Triplet batch loaded successfully:")
print(f"  Anchor shape: {a.shape}")
print(f"  Positive shape: {p.shape}")
print(f"  Negative shape: {n.shape}")

# ============================================================================
# DEFINE TRIPLET LOSS AND TRAINING SETUP
# ============================================================================
"""
PyTorch provides TripletMarginLoss:
Loss = max(||anchor - positive||² - ||anchor - negative||² + margin, 0)

For L2-normalized embeddings:
||a - b||² = 2 - 2(a·b) = 2(1 - cosine_similarity)

So minimizing Euclidean distance = maximizing cosine similarity
"""

print("\nSetting up training...")

# Define loss function
triplet_loss_fn = nn.TripletMarginLoss(
    margin=0.3,  # Minimum separation between positive and negative
    p=2          # Use L2 distance (Euclidean)
)

print(f"✓ Triplet loss defined:")
print(f"  Margin: 0.3")
print(f"  Distance metric: L2 (Euclidean)")

# Define optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,           # Learning rate
    weight_decay=1e-5  # L2 regularization
)

print(f"✓ Optimizer configured:")
print(f"  Type: Adam")
print(f"  Learning rate: 1e-4")
print(f"  Weight decay: 1e-5")

# ============================================================================
# TRAINING LOOP
# ============================================================================
"""
Train model for specified number of epochs.
"""

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        loader: Data loader (triplet)
        optimizer: Optimizer
        loss_fn: Loss function
        device: 'cuda' or 'cpu'
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    num_batches = len(loader)
    
    # Progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)
    
    for batch_idx, (anchor, positive, negative) in enumerate(pbar):
        # Move to device
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: Get embeddings for all three
        emb_anchor = model(anchor)      # (B, 128)
        emb_positive = model(positive)  # (B, 128)
        emb_negative = model(negative)  # (B, 128)
        
        # Compute triplet loss
        loss = loss_fn(emb_anchor, emb_positive, emb_negative)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Update progress bar
        if (batch_idx + 1) % 50 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate average loss
    avg_loss = running_loss / num_batches
    
    return avg_loss

# ============================================================================
# TRAIN FOR 5 EPOCHS
# ============================================================================

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

NUM_EPOCHS = 5
best_loss = float('inf')
save_path = '/mnt/user-data/outputs/mobilenet_triplet_vggface2.pt'

print(f"Training for {NUM_EPOCHS} epochs...")
print(f"Expected time: ~10 minutes on GPU\n")

# Track training history
train_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    # Train one epoch
    epoch_loss = train_one_epoch(
        model, triplet_train_loader, optimizer, 
        triplet_loss_fn, device, epoch
    )
    
    # Store loss
    train_losses.append(epoch_loss)
    
    print(f"\nEpoch {epoch}/{NUM_EPOCHS} - Avg Loss: {epoch_loss:.4f}")
    
    # Save if best model so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), save_path)
        print(f"✓ New best model saved (loss: {best_loss:.4f})")
    
    print("-" * 70)

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print(f"Best loss: {best_loss:.4f}")
print(f"Model saved to: {save_path}")
print(f"Training history: {train_losses}")

# Plot training curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Triplet Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/training_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Training curve saved to training_loss_curve.png")
print("="*70)
