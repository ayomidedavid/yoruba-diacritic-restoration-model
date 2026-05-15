Yoruba Diacritic Restoration — Hybrid BiLSTM + Transformer

This project provides a scaffold to train and evaluate models for restoring diacritics in Yoruba text. It includes:

- Character-level data preprocessing and dataset utilities (`src/data.py`)
- A Bi-LSTM sequence-to-sequence model (`src/models/bilstm.py`)
- A Transformer sequence-to-sequence model (`src/models/transformer.py`)
- A hybrid model combining Bi-LSTM + Transformer encoders (`src/models/hybrid.py`)
- Training scripts for each model (`src/train_bilstm.py`, `src/train_transformer.py`, `src/train_hybrid.py`)

Quick start (PowerShell):

```powershell
# create a venv and activate
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train a small BiLSTM model (example)
python .\src\train_bilstm.py --data_dir .\data --epochs 3 --batch_size 16
```

The code is intentionally minimal so you can adapt it to your full Yoruba dataset. See source files for details and extension points.

create 3 iptnb notebook 1 for bilstm 2nd for transfor mer and the last for the hybrid each notbook should contain all what is needed and it should use the main and all the dataset iw ant to go and run it on kaggle