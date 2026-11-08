"""Project paths shared by command-line entry points."""
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / 'data'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'


def new_output_path(category):
    """Each invocation gets a fresh location; never replace a previous run."""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    return OUTPUT_ROOT / category / timestamp


def create_output_folders(root, names):
    root.mkdir(parents=True, exist_ok=False)
    folders = {name: root / name for name in names}
    for folder in folders.values():
        folder.mkdir()
    return folders
