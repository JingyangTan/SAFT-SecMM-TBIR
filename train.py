# flake8: noqa: E402
import argparse
import json
import os
import sys

# Disable tokenizer parallelism to avoid deadlocks due to forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import src.datasets.retrieval_dataset as module_data
from src.config import validate_config

from src.trainers.trainer import RetrievalTrainer


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/train/tinyclip.json",
        type=str,
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="Path to a checkpoint used to resume training.",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers used by each data loader (default: 4).",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Random seed used by PyTorch Lightning (default: 1234).",
    )
    parser.add_argument(
        "--devices",
        default=[0],
        nargs="+",
        type=int,
        help="CUDA device indices passed to Lightning (default: 0).",
    )
    return parser.parse_args(args)


def load_config(config_path):
    with open(config_path, encoding="utf-8") as config_file:
        return json.load(config_file)


def build_dataset(config, train):
    dataset_config = config["dataset"]
    dataset_class = getattr(module_data, dataset_config["type"])
    return dataset_class(
        config["arch"]["args"]["model_arch"],
        config["arch"]["type"],
        train=train,
        **dataset_config["args"],
    )


def build_dataloaders(config, num_workers):
    train_dataset = build_dataset(config, train=True)
    val_dataset = build_dataset(config, train=False)
    batch_size = int(config["training"]["batch_size"])
    if len(train_dataset) < batch_size:
        raise ValueError(
            f"Training dataset has {len(train_dataset)} images, fewer than "
            f"batch_size={batch_size}; drop_last=True would produce no training batches."
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_dataloader, val_dataloader


def build_checkpoint_callback(config):
    return ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:0.6f}",
        every_n_train_steps=config["training"]["checkpoint_interval_steps"],
        save_on_train_epoch_end=False,
        save_top_k=10,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )


def build_logger(config):
    return TensorBoardLogger(
        save_dir="saved/" + config["name"],
        name=config["exp_name"],
    )


def build_trainer(config, logger, checkpoint_callback, max_epochs, devices):
    return pl.Trainer(
        logger=logger,
        devices=devices,
        max_epochs=max_epochs,
        val_check_interval=config["training"]["val_check_interval"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        deterministic=True,
    )


def save_json(path, content):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(content, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def save_run_metadata(config, args, logger):
    """Save the source config and command-line settings for this run."""
    os.makedirs(logger.log_dir, exist_ok=True)

    run_args = {
        "config_path": os.path.abspath(args.config),
        "resume": os.path.abspath(args.resume) if args.resume else None,
        "seed": args.seed,
        "devices": args.devices,
        "num_workers": args.num_workers,
        "command": [sys.executable, *sys.argv],
    }

    save_json(os.path.join(logger.log_dir, "source_config.json"), config)
    save_json(os.path.join(logger.log_dir, "run_args.json"), run_args)


def print_run_summary(args, config, max_epochs, logger):
    print("Training configuration:")
    print(f"  config: {args.config}")
    print(f"  seed: {args.seed}")
    print(f"  devices: {args.devices}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  batch_size: {config['training']['batch_size']}")
    print(f"  max_epochs: {max_epochs}")
    print(f"  resume: {args.resume or 'None'}")
    print(f"  log_dir: {logger.log_dir}")


def cli_main(args=None):
    args = parse_args(args)
    pl.seed_everything(args.seed)

    config = load_config(args.config)
    validate_config(config)
    max_epochs = config["training"]["max_epochs"]
    train_dataloader, val_dataloader = build_dataloaders(config, args.num_workers)
    model = RetrievalTrainer(config)

    logger = build_logger(config)
    checkpoint_callback = build_checkpoint_callback(config)
    save_run_metadata(config, args, logger)
    print_run_summary(args, config, max_epochs, logger)
    trainer = build_trainer(
        config, logger, checkpoint_callback, max_epochs, args.devices
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    cli_main()
