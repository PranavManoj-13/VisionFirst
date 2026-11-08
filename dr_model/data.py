import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

FOLDERS = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
PREPROCESSING = 'exif_transpose_rgb_bilinear_stretch_v2'


def load_records(root, return_audit=False):
    root = Path(root)
    records, seen = [], set()
    group_records = defaultdict(list)
    with (root / 'train.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            image_id, grade = row['id_code'], int(row['diagnosis'])
            if image_id in seen or not 0 <= grade <= 4 or Path(image_id).name != image_id:
                raise ValueError(f'Invalid or repeated row: {row}')
            seen.add(image_id)
            path = root / 'colored_images' / FOLDERS[grade] / (image_id + '.png')
            # Decode all images now: do not silently change the dataset on errors.
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            record = dict(id=image_id, grade=grade, path=str(path.resolve()), group=digest)
            records.append(record)
            group_records[digest].append(record)
    if not records:
        raise ValueError('Dataset is empty')
    conflicts = {key for key, rows in group_records.items() if len({r['grade'] for r in rows}) > 1}
    audit = {
        'source_images': len(records),
        'conflicting_groups_excluded': [
            {'group': key, 'images': [{'id': r['id'], 'grade': r['grade']} for r in group_records[key]]}
            for key in sorted(conflicts)
        ],
        'same_label_duplicate_groups': [
            {'group': key, 'ids': [r['id'] for r in rows]}
            for key, rows in group_records.items() if len(rows) > 1 and key not in conflicts
        ],
    }
    records = [r for r in records if r['group'] not in conflicts]
    audit['retained_images'] = len(records)
    return (records, audit) if return_audit else records


def split_records(records, seed=42):
    """Stratify independent pixel-identical groups, approximately 60/20/20.

    Group counts, rather than image counts, determine allocation. This prevents
    exact duplicate leakage; patient identities and near duplicates need a separate audit.
    """
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in ['train', 'validation', 'test']}
    for grade in range(5):
        groups = defaultdict(list)
        for row in records:
            if row['grade'] == grade:
                groups[row['group']].append(row)
        keys = sorted(groups)
        if len(keys) < 5:
            raise ValueError(f'Grade {grade} needs at least 5 independent groups')
        rng.shuffle(keys)
        n = max(1, round(len(keys) * 0.2))
        for name, subset in [('test', keys[:n]), ('validation', keys[n:2*n]), ('train', keys[2*n:])]:
            splits[name].extend(row for key in subset for row in groups[key])
    return splits


def load_rgb(path, size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        return image.copy()


class RetinalDataset(Dataset):
    def __init__(self, records, size=224, augment=False):
        self.records, self.size = records, size
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
        ]) if augment else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = load_rgb(row['path'], self.size)
        if self.augment:
            image = self.augment(image)
        return torch.from_numpy(np.array(image, dtype=np.float32)), row['grade']
