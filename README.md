# Amazon Product Recommender

A two-tower deep learning recommendation system that learns to predict Amazon products users will interact with based on historical review data. The model uses Bayesian Personalized Ranking (BPR) loss for pairwise learning with support for GPU acceleration via PyTorch.

## Overview

### What It Does

This project builds an end-to-end recommendation pipeline that:

1. **Loads and Preprocesses** Amazon review data (CSV)
2. **Maps IDs** from raw user/product identifiers to zero-indexed numeric values
3. **Prepares Data** with leave-k-out train/test splits and negative sampling
4. **Trains a Two-Tower Model** with separate user and item embedding towers
5. **Evaluates** using ranking metrics: Recall@10, Hit Rate@10, NDCG@10

### Architecture

**Two-Tower Model:**
- Separate embedding towers for users and products (64-dim embeddings by default)
- MLP layers for each tower to learn non-linear transformations
- Cosine similarity-based ranking during inference
- BPR loss during training to rank positive items above negative items

**Key Features:**
- ✅ GPU support (CUDA or CPU fallback)
- ✅ Mixed precision training (AMP)
- ✅ Efficient negative sampling with popularity-based weighting
- ✅ Configurable data preprocessing via JSON
- ✅ Full-rank evaluation with efficient batch processing

---

## How to Run

### Prerequisites

- **Python:** 3.11 - 3.14
- **Poetry:** For dependency management
- **CUDA (Optional):** For GPU acceleration
- **Dataset:** Amazon reviews CSV file

### Installation

1. **Clone or navigate to the project:**
   ```bash
   cd amazon-product-recommender
   ```

2. **Install dependencies with Poetry:**
   ```bash
   poetry install
   ```

3. **Verify PyTorch is installed:**
   ```bash
   poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   ```

### Configuration

Before running, ensure the configuration files are set up:

**`configs/dataloader.json`** - Specifies where to load the CSV data:
```json
{
    "data_path": "path/to/Reviews.csv",
    "file_type": "csv"
}
```

**`configs/dataprocessor.json`** - Defines preprocessing steps:
```json
{
    "use_cols": ["UserId", "ProductId", "Time", "Score", ...],
    "datetime_cols": ["Time"],
    "patterns": { ... },
    "fillna": { ... },
    "drop_duplicates": true
}
```

### Running the Pipeline

```bash
poetry run python -m amazon_product_recommender.main
```

**What happens:**

1. ✅ Loads CSV data
2. ✅ Preprocesses (cleans, normalizes, deduplicates)
3. ✅ Maps user/product IDs to indices
4. ✅ Splits into train/test (last 5 items per user → test)
5. ✅ Trains the two-tower model for 2 epochs
6. ✅ Evaluates on test set
7. ✅ Prints ranking metrics

**Expected Output:**
```
[2026-04-13 14:32:10] INFO - Using device: cuda
[2026-04-13 14:32:11] INFO - Preprocessing data with config
[2026-04-13 14:32:15] INFO - Mapping User & Product Ids to Indices
[2026-04-13 14:32:16] INFO - Splitting train/test
[2026-04-13 14:32:17] INFO - Initialize Negative Sampler
[2026-04-13 14:32:18] INFO - Creating PyTorch Dataset and initialize Pytorch DataLoader
[2026-04-13 14:32:19] INFO - Start training loop
Epoch 1: 100%|██████████| 1250/1250 [00:45<00:00, 27.8it/s]
Epoch 1/2 - Avg Loss: 0.153421
Epoch 2: 100%|██████████| 1250/1250 [00:45<00:00, 27.9it/s]
Epoch 2/2 - Avg Loss: 0.124156
[2026-04-13 14:33:05] INFO - Evaluating model
[2026-04-13 14:33:12] INFO - [RESULTS] {'recall@10': 0.23, 'hit_rate@10': 0.45, 'ndcg@10': 0.18}
```

### Adjusting Hyperparameters

Edit `amazon_product_recommender/main.py` to modify:

- **Training epochs:** `trainer.train(loader, epochs=2)`
- **Batch size:** `batch_size=256` in DataLoader
- **Number of negatives:** `num_negatives=5` in InteractionDataset
- **Embedding dim:** `embedding_dim=64` in TwoTowerModel
- **Learning rate:** `lr=1e-3` in Trainer initialization
- **Top-K evaluation:** `evaluator.evaluate(test_dict, k=10)`

### Project Structure

```
amazon-product-recommender/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── dataloader.json
│   └── dataprocessor.json
├── data/
│   └── Reviews.csv
├── amazon_product_recommender/
│   ├── main.py              # Entry point
│   ├── config.py            # Config loader
│   ├── data/
│   │   ├── dataloader.py    # CSV loading
│   │   ├── data_processor.py # Preprocessing
│   │   ├── id_mapper.py     # ID mapping to indices
│   │   ├── dataset.py       # PyTorch dataset wrapper
│   │   ├── samplers.py      # Negative sampling
│   │   └── train_test_split.py # Leave-k-out split
│   ├── models/
│   │   └── two_tower.py     # Two-tower model
│   ├── training/
│   │   ├── trainer.py       # Training loop
│   │   └── loss.py          # BPR loss function
│   └── evaluation/
│       └── evaluator.py     # Ranking metrics (Recall, Hit Rate, NDCG)
└── notebooks/
    └── test.ipynb           # Notebook for experimentation
```

---

## Troubleshooting

**Q: Out of memory error?**
A: Reduce `batch_size` in main.py (e.g., 64 or 128 instead of 256).

**Q: CPU is slow?**
A: Install PyTorch with CUDA support. Verify with `torch.cuda.is_available()`.

**Q: Config file not found?**
A: Ensure `configs/` directory exists at the project root with `.json` files inside.

**Q: Metrics are still tiny (1e-5)?**
A: Check that the `train()` method is being called and loss is decreasing during training.

---

## License

MIT License - See [LICENSE](LICENSE) for details.
