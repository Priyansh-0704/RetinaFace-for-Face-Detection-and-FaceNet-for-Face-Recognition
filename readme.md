# Face Recognition POC using RetinaFace and FaceNet

## Overview

This project implements a face recognition pipeline using:

- RetinaFace for face detection
- FaceNet for 128-dimensional embedding extraction
- Cosine similarity for identity matching

The system supports:
- Multi-face detection
- Known identity recognition
- Unknown face rejection
- Accuracy evaluation
- Performance measurement (CPU)


## Project Structure

```
retinaface-facenet-poc/

dataset/
    <identity_1>/
        img1.jpg
        img2.jpg
        ...
    <identity_2>/
    <identity_3>/
    ...

test_images/
    <identity_1>/
        img1.jpg
        img2.jpg
        ...
    <identity_2>/
    ...
    Unknown/

embeddings/   (generated file ignored in git)

generate_embeddings.py
recognize.py
evaluate.py
requirements.txt
.gitignore
README.md
```


## System Workflow

1. Detect faces using RetinaFace
2. Crop detected face region
3. Generate 128D embedding using FaceNet
4. Compare embeddings using cosine similarity
5. Assign identity based on threshold (0.6)


## Dataset Setup

- 5 identities
- 20 images per identity
- 15 images used for training
- 5 images used for testing
- Additional unknown faces used for open-set evaluation


## Installation

Step 1: Create virtual environment

python -m venv venv

Step 2: Activate environment (Windows)

venv\Scripts\activate

Step 3: Install dependencies

pip install -r requirements.txt


## Usage

Generate embeddings (run once after dataset setup):

python generate_embeddings.py

Run face recognition demo:

python recognize.py

Evaluate model accuracy:

python evaluate.py


## Example Results

Total Samples Evaluated: 29

Overall Accuracy: ~89.66%

Known Faces Accuracy: ~91.67%

Unknown Detection Accuracy: ~80%


Per Person Accuracy:

Brad_Pitt : 100.0 %

Christian_Bale : 100.0 %

Jane_Gyllenhaal : 100.0 %

Keanu Reeves : 75.0 %

Matt Damon : 80.0 %

Unknown : 80.0 %


## Performance Breakdown (CPU)

Average Detection Time : ~4.84 seconds

Average Embedding Time : ~1.25 seconds

Average Matching Time  : ~0.047 seconds

Average Total Time     : ~6.14 seconds 

---

## Author

Priyansh Agarwal  
Internship Project – Face Recognition (RetinaFace and FaceNet) POC
Argusoft Internship 2026