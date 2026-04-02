# Brain Tumor Segmentation using U-Net

## Overview

## Model
- U-Net
- BCE + Dice Loss
- Adam Optimizer

## Results
Dice Score: ~0.80

## Example Prediction
![Prediction](results/predictions.png)

## Project Structure
brain-tumor-unet/
│
├── data/
│   ├── images/          # MRI images
│   └── masks/           # Tumor masks
│
├── models/
│   └── unet.py          # U-Net architecture
│
├── dataset.py           # Dataset class
├── train.py             # Training script
├── eval.py              # Evaluation / prediction script
├── utils.py             # Loss + Dice score
│
├── results/
│   └── predictions.png  # Example predictions
│
├── README.md
├── requirements.txt
└── .gitignore
## How to Run
python train.py
python eval.py

## What I Learned

## Future Improvements