"""Download datasets from HuggingFace Hub."""

import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")

DATASETS = {
    "openwebmath": {
        "path": "open-web-math/open-web-math",
        "split": "train",
        "streaming": True,
        "max_samples": 100_000,
    },
    "numinamath_cot": {
        "path": "AI-MO/NuminaMath-CoT",
        "split": "train",
        "streaming": False,
        "max_samples": 50_000,
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "streaming": False,
        "max_samples": None,
    },
    "math": {
        "path": "hendrycks/competition_math",
        "split": "test",
        "streaming": False,
        "max_samples": None,
    },
}


def download_dataset(dataset_key: str):
    """Download a single dataset."""
    config = DATASETS[dataset_key]
    output_dir = RAW_DATA_DIR / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_key} from {config['path']}...")

    load_kwargs = {"path": config["path"], "split": config["split"]}
    if "name" in config:
        load_kwargs["name"] = config["name"]
    if config.get("streaming"):
        load_kwargs["streaming"] = True

    ds = load_dataset(**load_kwargs)

    if config.get("streaming"):
        samples = []
        for i, sample in enumerate(ds):
            if config["max_samples"] and i >= config["max_samples"]:
                break
            samples.append(sample)

        from datasets import Dataset
        ds = Dataset.from_list(samples)
    elif config.get("max_samples"):
        ds = ds.select(range(min(len(ds), config["max_samples"])))

    ds.save_to_disk(str(output_dir))
    logger.info(f"Saved {len(ds)} samples to {output_dir}")
    return ds


def download_all():
    """Download all datasets."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for key in DATASETS:
        try:
            download_dataset(key)
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")


if __name__ == "__main__":
    download_all()
