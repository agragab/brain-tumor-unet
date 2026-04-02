# Brain Tumor Segmentation using U-Net

This project implements a U-Net model in PyTorch for brain tumor segmentation from MRI scans.

## Project Structure

brain-tumor-unet/
├── data/
├── models/
│   └── unet.py
├── dataset.py
├── train.py
├── eval.py
├── utils.py
├── results/
│   └── predictions.png
├── README.md
└── requirements.txt

## Features
- U-Net architecture implemented from scratch
- Custom PyTorch Dataset
- Training and validation loops
- Dice score evaluation
- Prediction visualization

## Results
The model achieved a validation Dice score of around 0.80.

## Run

Train:
```bash
python train.py