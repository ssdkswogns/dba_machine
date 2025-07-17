import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class UAHDataset(Dataset):
    """
    Dataset for sliding-window .npz files created by segment_and_save.
    Uses metadata.json and splits.json to select a split.

    Args:
        root_dir (str): directory containing .npz files, metadata.json, splits.json
        split_name (str): one of the keys in splits.json (e.g., 'train', 'val', 'test')
        metadata_file (str): name of metadata file (default 'metadata.json')
        splits_file (str): name of splits file (default 'splits.json')
    """
    def __init__(self, root_dir, split_name='train', metadata_file='metadata.json', splits_file='splits.json', mean=None, std=None):
        self.root = root_dir
        # load metadata
        meta_path = os.path.join(root_dir, metadata_file)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        # load splits
        splits_path = os.path.join(root_dir, splits_file)
        if not os.path.isfile(splits_path):
            raise FileNotFoundError(f"Splits file not found: {splits_path}")
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        if split_name not in splits:
            raise ValueError(f"Split '{split_name}' not found in {splits_file}")
        valid_set = set(splits[split_name])
        # filter metadata to only include files in the desired split
        self.samples = [item for item in metadata if item['file'] in valid_set]
        if not self.samples:
            raise ValueError(f"No samples found for split '{split_name}'")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        npz_path = os.path.join(self.root, entry['file'])
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}")
        arr = np.load(npz_path)
        # retrieve features and label
        feats = arr['feats'] if 'feats' in arr else arr[arr.files[0]]
        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean)
            std = np.array(self.std)
            feats = (feats - mean) / (std + 1e-8)
        label = int(arr['label']) if 'label' in arr else int(entry.get('label', 0))
        return torch.from_numpy(feats).float(), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    Collate variable-length sequence batches by padding to the max length.
    Returns: padded_feats (B, L_max, D), labels (B,), lengths (B,)
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    padded = pad_sequence(xs, batch_first=True)
    labels = torch.stack(ys)
    return padded, labels

def compute_mean_std(dataset):
    # dataset: train split UAHDataset
    all_feats = []
    for feats, _ in dataset:
        all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0)  # (N, D)
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0)
    return mean, std