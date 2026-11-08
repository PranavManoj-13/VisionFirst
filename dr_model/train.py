"""Train RetinaScan v2. See README.md; old .pth files are incompatible."""
import argparse
import csv
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from dr_model.data import PREPROCESSING, RetinalDataset, load_records, split_records
from dr_model.paths import DATA_ROOT, create_output_folders, new_output_path
from dr_model.metrics import metrics
from dr_model.model import ARCHITECTURE, RetinalModel, ordinal_loss


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def evaluate(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(targets)
    logits, labels = torch.cat(logits), torch.cat(labels)
    return logits, labels, float(ordinal_loss(logits, labels))


def fit_temperature(logits, labels):
    """Validation-only NLL calibration, positive bounded temperature; never use test."""
    # A bounded one-dimensional search is robust and avoids negative temperatures.
    candidates = torch.logspace(-1, 1, 201)
    losses = [float(ordinal_loss(logits / t, labels)) for t in candidates]
    index = int(np.argmin(losses))
    return float(candidates[index]), {
        'method': 'validation ordinal BCE grid search', 'range': [0.1, 10.0],
        'before_bce': float(ordinal_loss(logits, labels)),
        'after_bce': losses[index], 'at_search_boundary': index in (0, len(losses)-1)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path, default=None, help='New run directory; default: outputs/runs/<UTC timestamp>')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224, choices=[224, 320, 384])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--no-pretrained', action='store_true', help='For smoke tests only')
    parser.add_argument('--validate-only', action='store_true', help='Audit data and save splits without training')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0 or args.workers < 0:
        parser.error('Epochs, batch size, patience, and LR must be positive; workers nonnegative')
    if args.output is None:
        args.output = new_output_path('runs')
    seed_everything(args.seed)
    records, audit = load_records(args.data_root, return_audit=True)
    splits = split_records(records, args.seed)
    folders = create_output_folders(args.output, ['checkpoints', 'metadata', 'logs', 'metrics', 'predictions'])
    write_json(folders['metadata'] / 'data_audit.json', audit)
    print('Excluded conflicting pixel groups:', len(audit['conflicting_groups_excluded']), flush=True)
    config = {**vars(args), 'data_root': str(args.data_root.resolve()), 'output': str(args.output.resolve())}
    fingerprint = hashlib.sha256(json.dumps(sorted((r['id'], r['grade'], r['group']) for r in records)).encode()).hexdigest()
    write_json(folders['metadata'] / 'run.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'architecture': ARCHITECTURE, 'preprocessing': PREPROCESSING,
        'dataset_sha256': fingerprint, 'python': platform.python_version(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__,
        'split_counts': {k: len(v) for k, v in splits.items()},
        'unique_pixel_groups': len({r['group'] for r in records}),
        'limitation': 'No patient IDs supplied; pixel grouping does not detect patient or near-duplicate leakage.'
    })
    with (folders['metadata'] / 'splits.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'id', 'grade', 'path', 'group'])
        writer.writeheader()
        for name, rows in splits.items():
            writer.writerows(dict(split=name, **r) for r in rows)
    print('Splits:', {k: len(v) for k, v in splits.items()}, flush=True)
    if args.validate_only:
        return
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device_name)
    print(f'Device: {device} | Run directory: {args.output.resolve()}', flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(
        RetinalDataset(rows, args.image_size, augment=name == 'train'),
        batch_size=args.batch_size, shuffle=name == 'train', num_workers=args.workers,
        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=generator
    ) for name, rows in splits.items()}
    model = RetinalModel(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_qwk, stale, history = -float('inf'), 0, []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ordinal_loss(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        logits, labels, val_loss = evaluate(model, loaders['validation'], device)
        report = metrics(labels.numpy(), logits.sigmoid().numpy())
        score = report['qwk']
        if score is None or not np.isfinite(score):
            raise RuntimeError('Validation QWK is undefined')
        history.append(dict(epoch=epoch+1, train_loss=total/len(splits['train']),
                            val_loss=val_loss, val_qwk=score, lr=optimizer.param_groups[0]['lr']))
        write_json(folders['logs'] / 'history.json', history)
        print(history[-1], flush=True)
        if score > best_qwk + 1e-5:
            best_qwk, stale = score, 0
            torch.save({
                'format_version': 2, 'architecture': ARCHITECTURE, 'preprocessing': PREPROCESSING,
                'image_size': args.image_size, 'state_dict': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'epoch': epoch+1, 'validation_qwk': score, 'dataset_sha256': fingerprint,
                'temperature': 1.0, 'config': config,
            }, folders['checkpoints'] / 'best.pt')
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    checkpoint = torch.load(folders['checkpoints'] / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    logits, labels, _ = evaluate(model, loaders['validation'], device)
    temperature, calibration = fit_temperature(logits, labels)
    checkpoint['temperature'] = temperature
    torch.save(checkpoint, folders['checkpoints'] / 'best.pt')
    write_json(folders['metrics'] / 'calibration.json', dict(temperature=temperature, **calibration))
    # Test is first used here, after architecture/training/selection/calibration are complete.
    logits, labels, _ = evaluate(model, loaders['test'], device)
    cumulative = (logits / temperature).sigmoid().numpy()
    write_json(folders['metrics'] / 'test_metrics.json', metrics(labels.numpy(), cumulative))
    with (folders['predictions'] / 'test_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'true_grade', 'predicted_grade', 'p_gt_0', 'p_gt_1', 'p_gt_2', 'p_gt_3'])
        for row, probs in zip(splits['test'], cumulative):
            writer.writerow([row['id'], row['grade'], int((probs > .5).sum()), *probs.tolist()])
    print('Saved checkpoints/best.pt and measured test results to', args.output, flush=True)


if __name__ == '__main__':
    main()
