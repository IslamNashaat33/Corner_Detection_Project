# ML Project

## Description
This project contains scripts and models for generating, processing, and evaluating datasets for machine learning tasks. It includes:

- **Dataset Creation**: Scripts to generate synthetic datasets with labeled strokes and corners.
- **Dataset Splitting**: Scripts to split datasets into training, validation, and test sets.
- **Neural Network Model**: A BiLSTM model for corner detection in strokes.
- **Evaluation**: Scripts to evaluate the model's performance.

## Folder Structure
```
ML Project/
├── dataset/                # Raw dataset files
├── dataset creation/       # Scripts for dataset generation and splitting
├── dataset_clean/          # Cleaned dataset
├── dataset_split/          # Split dataset (train/val/test)
├── NN model/               # Neural network model and evaluation scripts
├── DT model/               # Decision tree model scripts
```

## Usage

### 1. Generate Dataset
Run the dataset generation script:
```bash
python dataset creation/intersection_dataset_v2.py
```

### 2. Split Dataset
Split the dataset into train/val/test:
```bash
python dataset creation/split_dataset.py
```

### 3. Train Neural Network
Train the BiLSTM model:
```bash
python NN model/train.py
```

### 4. Evaluate Model
Evaluate the trained model:
```bash
python NN model/evaluate_test.py
```

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.