"""Main entry point for the Amazon product recommender pipeline.

Orchestrates the complete workflow: data loading, preprocessing, ID mapping,
sampling, dataset creation, model initialization, and training.
"""

import logging

import torch
from config import Config
from data.dataloader import DataLoader
from data.dataset import AmazonDataset
from data.id_mapper import IdMapper
from data.preprocessing.amazon_preprocessor import AmazonPreProcessor
from data.text_embedding import ProductEmbeddingBuilder, ProductTextAggregator, TextEncoder
from data.train_test_split import train_test_split
from data.utils import collate_fn, load_embeddings, set_seed
from evaluation.evaluator import Evaluator
from models.two_tower import TwoTowerModel
from torch.utils.data import DataLoader as TorchDataLoader
from training.loss import BPRLoss
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    """Run full training pipeline for two-tower recommender system."""
    config_settings = Config("settings.json")
    set_seed(config_settings.get("seed", 42))

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
    dp = AmazonPreProcessor(df, config_dp)
    df = dp.preprocess()

    # Create review column by combining summary and text
    df = dp.make_review_text(summary_weight=config_settings.get("summary_weight", 1))

    # -----------------------------
    # 3. Train / Test split
    # -----------------------------
    logging.info("Splitting train/test")
    config_split = config_settings.get("train_test_split", {})
    train_df, test_df = train_test_split(
        df,
        strategy=config_split.get("strategy", "leave_k_out"),
        k=config_split.get("k", 5),
        datetime_col=config_split.get("datetime_col", "time"),
        threshold=config_split.get("threshold", 0.8),
    )

    # -----------------------------
    # 4. Map IDs → indices & Build user-products dict
    # -----------------------------
    logging.info("Mapping User & Product Ids to Indices")
    mapper = IdMapper()
    mapper.fit(train_df)
    train_df = mapper.transform(train_df)
    test_df = mapper.transform(test_df)

    # -----------------------------
    # 5. Aggregate product texts
    # -----------------------------
    logging.info("Aggregating product review texts")
    config_aggregator = config_settings.get("text_aggregator", {})
    aggregator = ProductTextAggregator(
        max_reviews=config_aggregator.get("max_reviews", 20), max_chars=config_aggregator.get("max_chars", 2000)
    )

    product_texts = aggregator.aggregate(train_df)

    # -----------------------------
    # 6. Build review text embeddings using transformers
    # -----------------------------
    logging.info("Building product embeddings")
    model_name = config_settings.get(
        "encoder_name", "sentence-transformers/all-MiniLM-L6-v2"
    )  # "bert-base-uncased" or "sentence-transformers/all-MiniLM-L6-v2"

    logging.info(f"Using encoder: {model_name}")
    encoder = TextEncoder(model_name=model_name, device=device)

    config_embeddings = config_settings.get("product_embeddings_paths", {})
    if not config_embeddings.get("load_path", None):
        builder = ProductEmbeddingBuilder(encoder)
        embedding_matrix, product_ids = builder.build(
            product_texts, save_path=config_embeddings.get("save_path", "product_embeddings.pt")
        )
    else:
        embedding_matrix, product_ids = load_embeddings(config_embeddings.get("load_path"), device=device)

    # -----------------------------
    # 7. Dataset + DataLoader
    # -----------------------------
    logging.info("Creating PyTorch Dataset and initialize Pytorch DataLoader")
    # Build train interaction dict (for masking)
    train_user_product_dict = train_df.groupby("user_idx")["product_idx"].apply(set).to_dict()

    train_dataset = AmazonDataset(
        config=config_settings.get("config_dataset", {}),
        df=train_df,
        embedding_matrix=embedding_matrix,
        product_ids=product_ids,
        user_product_dict=train_user_product_dict,
    )

    config_trainer = config_settings.get("trainer", {})
    train_loader = TorchDataLoader(
        dataset=train_dataset,
        batch_size=config_trainer.get("batch_size", 256),
        shuffle=config_trainer.get("shuffle", True),
        collate_fn=collate_fn,
        num_workers=config_trainer.get("num_workers", 2),
        pin_memory=True,
    )

    # -----------------------------
    # 8. TwoTowerModel
    # -----------------------------
    logging.info("Initializing TwoTowerModel")
    embedding_dim = embedding_matrix.shape[1]

    model = TwoTowerModel(embedding_dim=embedding_dim, hidden_dim=config_settings.get("hidden_dim", 256)).to(device)

    # -----------------------------
    # 9. Trainer
    # -----------------------------
    logging.info("Start training loop")
    loss_fn = BPRLoss(reg_weight=config_trainer.get("loss_reg_weights", 1e-4))
    trainer = Trainer(model=model, loss_fn=loss_fn, device=device, lr=config_trainer.get("learning_rate", 1e-3))

    trainer.train(train_loader, epochs=config_trainer.get("epochs", 10))

    # -----------------------------
    # 10. Train Evaluator
    # -----------------------------
    logging.info("Start evaluation")

    # Since we only have embeddings for products in the training set, we need to filter the test set to those products.
    valid_products = set(product_ids)
    valid_users = set(mapper.user2idx.values())
    test_df = test_df[test_df["product_idx"].isin(valid_products) & test_df["user_idx"].isin(valid_users)]

    test_user_product_dict = test_df.groupby("user_idx")["product_idx"].apply(set).to_dict()

    evaluator = Evaluator(
        model=model,
        item_embeddings=embedding_matrix,
        product_ids=product_ids,
        user_product_dict=train_user_product_dict,
        device=device,
    )

    metrics = evaluator.evaluate(
        test_dict=test_user_product_dict,
        k=config_trainer.get("top_k_recommendations", 10),
        batch_size=config_trainer.get("batch_size", 256),
    )
    logging.info(f"Evaluation Metrics: {metrics}")


if __name__ == "__main__":
    main()
