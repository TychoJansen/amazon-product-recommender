"""Main entry point for the Amazon product recommender pipeline.

Orchestrates the complete workflow: data loading, preprocessing, ID mapping,
sampling, dataset creation, model initialization, and training.
"""

import logging
import random

import numpy as np
import torch
from config import Config
from data.data_processor import DataProcessor
from data.dataloader import DataLoader
from data.dataset import InteractionDataset
from data.id_mapper import IdMapper
from data.samplers import NegativeSampler
from data.train_test_split import train_test_split
from evaluation.evaluator import Evaluator
from models.two_tower import TwoTowerModel
from torch.utils.data import DataLoader as TorchDataLoader
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_seed(seed: int = 42):
    """Set random seeds for reproductibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Run full training pipeline for two-tower recommender system."""
    set_seed(42)

    # -----------------------------
    # 0. Device setup (GPU support)
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # -----------------------------
    # 1. Load raw dataset
    # -----------------------------
    config_dl = Config("dataloader.json")
    dl = DataLoader(config_dl)
    df = dl.load().get_data()

    # -----------------------------
    # 2. Preprocess dataset
    # -----------------------------
    logging.info("Preprocessing data with config")
    config_dp = Config("dataprocessor.json")
    dp = DataProcessor(df, config_dp)
    df = dp.preprocess()

    # -----------------------------
    # 3. Map IDs → indices & Build user-products dict
    # -----------------------------
    logging.info("Mapping User & Product Ids to Indices")
    mapper = IdMapper()
    mapper.fit(df)
    df = mapper.transform(df)

    # -----------------------------
    # 4. Train / Test split
    # -----------------------------
    logging.info("Splitting train/test")
    train_df, test_dict = train_test_split(df, strategy="leave_k_out", k=5, datetime_col="time", threshold=0.8)
    user_product_dict = mapper.build_user_product_dict(train_df)

    breakpoint()

    logging.info("Initialize Negative Sampler")

    # Calulute item popularity for sampling
    num_users = int(df["user_idx"].max()) + 1
    num_products = int(df["product_idx"].max()) + 1

    popularity = np.bincount(train_df["product_idx"].values, minlength=num_products)

    sampler = NegativeSampler(
        num_items=num_products,
        user_product_dict=user_product_dict,
        popularity=popularity,
    )

    # -----------------------------
    # 6. Dataset + DataLoader
    # -----------------------------
    logging.info("Creating PyTorch Dataset and initialize Pytorch DataLoader")
    dataset = InteractionDataset(
        train_df["user_idx"].values, train_df["product_idx"].values, sampler=sampler, num_negatives=5
    )

    loader = TorchDataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # -----------------------------
    # 7. Model
    # -----------------------------
    model = TwoTowerModel(
        num_users=num_users,
        num_products=num_products,
    )

    # -----------------------------
    # 8. Trainer
    # -----------------------------
    logging.info("Start training loop")
    trainer = Trainer(model=model, device=device)
    trainer.train(loader, epochs=2)

    # -----------------------------
    # 8. Build item embeddings for
    # -----------------------------
    logging.info("Building item embeddings")

    with torch.no_grad():
        item_ids = torch.arange(num_products).to(device)
        item_embeddings = model.encode_items(item_ids)

    # -----------------------------
    # 8. Evaluation
    # -----------------------------
    logging.info("Evaluating model")

    evaluator = Evaluator(
        model=model,
        num_items=num_products,
        user_product_dict=user_product_dict,
        item_embeddings=item_embeddings,
        device=device,
    )

    metrics = evaluator.evaluate(test_dict, k=10)
    logging.info(f"[RESULTS] {metrics}")


if __name__ == "__main__":
    main()
