# LY-Project

Churn Prediction + RMAB (Restless Multi-Armed Bandit) intervention pipeline.

## Summary
- Preprocess raw CSVs -> `processed/processed.csv`
- Split into train/test (80:20) -> `processed/train.csv`, `processed/test.csv`
- Train PyTorch MLP model (uses MPS on Mac M1/M2 if available)
- Evaluate with SMAPE, F1, accuracy, precision, recall, ROC-AUC
- Plot train vs validation loss -> `outputs/loss_curve.png`
- Simulate RMAB policy using approximate Whittle index -> `outputs/rmab_simulation.json`

## Quick commands
```bash
# create venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# preprocess, split, train
python3 preprocess.py
python3 split_dataset.py
bash run_train.sh

# run RMAB simulation (after training)
python3 rmab_run.py
