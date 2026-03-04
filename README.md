# Plant Disease Detection System

A modular TensorFlow project for detecting diseases in various plants using MobileNetV2.

## Supported Models
- Crop Classifier
- Potato Disease Model
- Onion Disease Model
- Apple Disease Model
- Tomato Disease Model

## Structure
- `src/`: Core logic (preprocessing, loading, building, evaluation, prediction).
- `models/`: Saved `.h5` models and label maps.
- `train.py`: Main entry for training.

## Usage
1. Setup environmental: `py -3.12 -m venv venv` and `pip install -r requirements.txt`.
2. Organize your dataset in `dataset/<model_name>/`.
3. Train: `python train.py --dataset ./dataset/potato --model_name potato_model`
