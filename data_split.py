import os
import json
import argparse
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser("Sliding Window Data Split")
    parser.add_argument("--meta", type=str, default="./data/UAH_dataset_processed/metadata.json",
                        help="Path to metadata.json for windowed dataset")
    parser.add_argument("--output", type=str, default="./data/splits.json",
                        help="Output JSON file for splits")
    parser.add_argument("--val-size", type=float, default=0.3,
                        help="Fraction of data for validation set")
    parser.add_argument("--test-size", type=float, default=0,
                        help="Fraction of data for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Load metadata
    with open(args.meta, 'r') as f:
        meta = json.load(f)
    train_files = [item['file'] for item in meta]
    train_labels = [item['label'] for item in meta]

    print(f"train_labels: {train_labels}")

    if args.test_size > 0:
        # Split off test set
        train_files, test_files, train_labels, test_labels = train_test_split(
            train_files, train_labels,
            test_size=args.test_size,
            stratify=train_labels,
            random_state=args.seed
        )
    # Split train and validation
    rel_val = args.val_size / (1.0 - args.test_size)
    train_files, val_files, _, _ = train_test_split(
        train_files, train_labels,
        test_size=rel_val,
        stratify=train_labels,
        random_state=args.seed
    )
    # Save splits
    splits = {
        'train': train_files,
        'val': val_files,
    }
    with open(args.output, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved sliding-window splits to {args.output}")