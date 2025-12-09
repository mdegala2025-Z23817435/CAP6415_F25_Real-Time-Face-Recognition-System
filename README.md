# Facial Embedding Learning and Gender Classification Using Triplet Networks on VGGFace2

## Abstract<br>
This project implements a complete face-representation learning system using the VGGFace2 dataset and a MobileNet-based Siamese/Triplet architecture.

##The goal is to solve two core problems:<br>
1.	Learn discriminative face embeddings capable of identifying individuals using cosine similarity.
2.	Build a scalable embedding-database system which:
    2.1 Stores embeddings for all known faces
    2.2 Retrieves top-K similar faces
    2.3 Automatically adds unseen faces without retraining

Additionally, a lightweight gender classifier is built on top of the learned embeddings, achieving 97.6% accuracy and an AUC of 0.996, demonstrating the strength of the learned representations.
The project includes full training scripts, evaluation tools, visualizations, reproducibility guarantees, and a complete inference pipeline.

Pipeline Overview
1. Dataset Preparation (VGGFace2)
•	Download via KaggleHub
•	Filter identities with ≥20 images
•	Clean corrupted files
•	Split into train (80%) and val (20%)
•	Apply augmentation + normalization

2. Embedding Model – Triplet Learning
•	Backbone: MobileNetV2
•	Output: 128-dim normalized embeddings
•	Loss: TripletMarginLoss, margin = 0.3
•	Optimizer: Adam
•	5 training epochs
•	Best checkpoint saved as mobilenet_triplet_vggface2.pt

3. Identity Evaluation
•	Build gallery using 1 image per identity
•	Compute embeddings for gallery & probes
•	Identify using nearest-neighbor cosine similarity
•	Achieved 81.21% identification accuracy


4. Embedding Database System
•	Stores embeddings for all images
•	Query system:
o	Computes embedding for new image
o	Returns top-K similar images
o	If similarity < threshold → marks as new person and adds to DB

5. Gender Classification Layer
•	Freeze embedding model
•	Add a 128 → 2 classifier head
•	Train on gender-labelled subset
•	Achieved:
o	97.6% accuracy
o	AUC = 0.996
o	Excellent class separation

6. Inference Pipeline
→ Load & preprocess → Generate embedding → Predict gender + show graph → Retrieve similar images from DB → Add new faces automatically
Visual Results
Confusion Matrix and ROC Curve
AUC = 0.996, nearly perfect.
 
Gender Prediction Example
 

 

Reproducibility
To reproduce:
pip install -r requirements.txt
python src/triplet_training.py
python src/gender_classifier.py
python src/build_embedding_db.py
python src/inference.py

The code is deterministic:
•	Seed = 42
•	CUDNN deterministic flags set

Hardware used: NVIDIA L4 GPU 
Results from the TA machine should match closely (±1% accuracy).
Folder Descriptions
src/
Contains modularized code:
•	dataset_loader.py → load VGGFace2 subset
•	embedding_model.py → MobileNet embedding network
•	triplet_training.py → Triplet loss training
•	gender_classifier.py → Gender model & metrics
•	embedding_database.py → Similarity search + auto-enrolment
•	inference.py → Combined pipeline

logs/
Contains weekly development logs.

results/
Contains:
•	Accuracy metrics
•	ROC curve
•	Confusion matrix
•	Retrieval screenshots
