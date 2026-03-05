"""
ZINC Dataset Download Script
============================
Downloads a subset of the ZINC dataset from Hugging Face and saves it
locally as a parquet file for reproducible offline use.

Usage:
    python scripts/download_zinc.py

Dataset:
    HF name : graphs-datasets/ZINC
    Split   : train (first 50k rows by default)
    Columns : smiles (SMILES string)

After successful download the file is saved to:
    data/zinc_50k.parquet
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def download(num_rows: int = 50_000, out_dir: Path = Path("data")) -> Path:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install dependencies first: pip install datasets pyarrow")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zinc_{num_rows // 1000}k.parquet"

    if out_path.exists():
        log.info("Dataset already exists at %s, skipping download.", out_path)
        return out_path

    log.info("Downloading %d rows from graphs-datasets/ZINC …", num_rows)
    ds = load_dataset(
        "graphs-datasets/ZINC",
        split=f"train[:{num_rows}]",
        trust_remote_code=False,
    )
    log.info("Downloaded %d rows. Columns: %s", len(ds), ds.column_names)

    # graphs-datasets/ZINC may store SMILES in 'smiles' or 'smi'
    smiles_col = next((c for c in ("smiles", "smi", "SMILES") if c in ds.column_names), ds.column_names[0])
    log.info("Using SMILES column: '%s'", smiles_col)
    ds = ds.select_columns([smiles_col])
    if smiles_col != "smiles":
        ds = ds.rename_column(smiles_col, "smiles")

    ds.to_parquet(str(out_path))
    log.info("Saved to %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ZINC dataset slice from Hugging Face")
    parser.add_argument("--rows", type=int, default=50_000, help="Number of SMILES rows to download")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    download(args.rows, Path(args.out_dir))
