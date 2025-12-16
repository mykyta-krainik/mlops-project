import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def split_dataset(
    input_file: Path,
    output_dir: Path,
    num_batches: int = 10,
    shuffle: bool = True,
    random_state: int = 255,
) -> list[Path]:
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = len(df) // num_batches
    remainder = len(df) % num_batches

    batch_files = []
    start_idx = 0

    for i in tqdm(range(num_batches)):
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size

        batch_df = df.iloc[start_idx:end_idx]

        batch_file = output_dir / f"batch_{i:03d}.csv"
        batch_df.to_csv(batch_file, index=False)
        batch_files.append(batch_file)

        start_idx = end_idx

    return batch_files


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into batches for streaming simulation"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("jigsaw-toxic-comment-classification-challenge/train.csv"),
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/batches"),
        help="Output directory for batch files",
    )
    parser.add_argument(
        "--batches",
        "-n",
        type=int,
        default=10,
        help="Number of batches to create",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle data before splitting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=255,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    batch_files = split_dataset(
        input_file=args.input,
        output_dir=args.output,
        num_batches=args.batches,
        shuffle=not args.no_shuffle,
        random_state=args.seed,
    )

    print("\nBatch summary:")
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        print(f"  {batch_file.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
