# Facial Embedding Learning and Gender Classification Using Triplet Networks on VGGFace2

## Abstract<br>
This project implements a complete face-representation learning system using the VGGFace2 dataset and a MobileNet-based Siamese/Triplet architecture.

## The goal is to solve two core problems:<br>
1.	Learn discriminative face embeddings capable of identifying individuals using cosine similarity.<br>
2.	Build a scalable embedding-database system which:<br>
    2.1 Stores embeddings for all known faces<br>
    2.2 Retrieves top-K similar faces<br>
    2.3 Automatically adds unseen faces without retraining<br>

Additionally, a lightweight gender classifier is built on top of the learned embeddings, achieving 97.6% accuracy and an AUC of 0.996, demonstrating the strength of the learned representations.<br>
The project includes full training scripts, evaluation tools, visualizations, reproducibility guarantees, and a complete inference pipeline.<br>

## Pipeline Overview<br>

## 1. Dataset Preparation (VGGFace2)<br>
•	Download via KaggleHub<br>
•	Filter identities with ≥20 images<br>
•	Clean corrupted files<br>
•	Split into train (80%) and val (20%)<br>
•	Apply augmentation + normalization<br>

## 2. Embedding Model – Triplet Learning<br>
•	Backbone: MobileNetV2<br>
•	Output: 128-dim normalized embeddings<br>
•	Loss: TripletMarginLoss, margin = 0.3<br>
•	Optimizer: Adam<br>
•	5 training epochs<br>
•	Best checkpoint saved as mobilenet_triplet_vggface2.pt<br>

## 3. Identity Evaluation<br>
•	Build gallery using 1 image per identity<br>
•	Compute embeddings for gallery & probes<br>
•	Identify using nearest-neighbor cosine similarity<br>
•	Achieved 81.21% identification accuracy<br>

## 4. Embedding Database System<br>
•	Stores embeddings for all images<br>
•	Query system:<br>
    o Computes embedding for new image<br>
    o Returns top-K similar images<br>
    o If similarity < threshold → marks as new person and adds to DB<br>

## 5. Gender Classification Layer<br>
•	Freeze embedding model<br>
•	Add a 128 → 2 classifier head<br>
•	Train on gender-labelled subset<br>
•	Achieved:<br>
    o 97.6% accuracy<br>
    o AUC = 0.996<br>
    o Excellent class separation<br>

## 6. Inference Pipeline<br>
→ Load & preprocess → Generate embedding → Predict gender + show graph → Retrieve similar images from DB → Add new faces automatically<br>

# Visual Results
## -----------------
## Confusion Matrix and ROC Curve<br>
AUC = 0.996, nearly perfect.<br>
<img width="940" height="380" alt="image" src="https://github.com/user-attachments/assets/83658303-8544-41c9-b75d-77567d52643a" /><br>

Gender Prediction Example<br>
<img width="582" height="313" alt="image" src="https://github.com/user-attachments/assets/1c6f7c44-185c-4c8f-ad3f-a3771d9ef2d1" /><br>
<img width="612" height="323" alt="image" src="https://github.com/user-attachments/assets/6375a2d1-9bbd-400a-9788-03da26f12368" /><br>
<img width="582" height="315" alt="image" src="https://github.com/user-attachments/assets/5c5fddcd-a93a-4efe-93e6-36b2d70e32af" /><br>

## Reproducibility<br>
To reproduce:<br>
    o pip install -r requirements.txt<br>
    o python src/triplet_training.py<br>
    o python src/gender_classifier.py<br>
    o python src/build_embedding_db.py<br>
    o python src/inference.py<br>

## The code is deterministic:<br>
•	Seed = 42<br>
•	CUDNN deterministic flags set<br>

## Hardware used: NVIDIA L4 GPU<br>

## Folder Descriptions<br>
## src/<br>
Contains modularized code:<br>
•	dataset_loader.py → load VGGFace2 subset<br>
•	embedding_model.py → MobileNet embedding network<br>
•	triplet_training.py → Triplet loss training<br>
•	gender_classifier.py → Gender model & metrics<br>
•	embedding_database.py → Similarity search + auto-enrolment<br>
•	inference.py → Combined pipeline<br>

## logs/<br>
Contains weekly development logs.<br>

## results/ - Contains:<br>
•	Accuracy metrics<br>
•	ROC curve<br>
•	Confusion matrix
•	Retrieval screenshots
