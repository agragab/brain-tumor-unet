# Brain Tumor Segmentation using U-Net

This project implements a U-Net model in PyTorch for brain tumor segmentation from MRI scans.

<pre> ## Project Structure ``` brain-tumor-unet/ │ ├── data/ │ ├── images/ │ └── masks/ │ ├── models/ │ └── unet.py │ ├── dataset.py ├── train.py ├── eval.py ├── utils.py │ ├── results/ │ └── predictions.png │ ├── README.md ├── requirements.txt └── .gitignore ``` </pre>

## Features
- U-Net architecture implemented from scratch
- Custom PyTorch Dataset
- Training and validation loops
- Dice score evaluation
- Prediction visualization

## Results
The model achieved a validation Dice score of around 0.80.

## Example Prediction

Below is an example of the model's tumor segmentation:

![Prediction](results/predictions.png)

## Run

Train:
```bash
python train.py