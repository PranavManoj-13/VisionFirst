"""Export a v2 checkpoint and validate the exported runtime on held-out fixtures."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch

from dr_model.paths import create_output_folders, new_output_path
from dr_model.data import PREPROCESSING, load_rgb
from dr_model.model import ARCHITECTURE, LABELS, InferenceModel, RetinalModel


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 2 or checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Requires a newly trained v2 checkpoint, not an old fold checkpoint')
    if checkpoint.get('preprocessing') != PREPROCESSING:
        raise ValueError('Unknown preprocessing contract')
    temperature = checkpoint['temperature']
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid calibration temperature')
    model = RetinalModel(pretrained=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return InferenceModel(model.eval(), temperature).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=None, help='New export directory; default: outputs/exports/<UTC timestamp>')
    parser.add_argument('--format', choices=['tflite', 'torchscript'], default='tflite')
    parser.add_argument('--images', type=Path, nargs='+', required=True, help='Real images for conversion parity checks')
    args = parser.parse_args()
    if args.output is None:
        args.output = new_output_path('exports')
    model, checkpoint = load_checkpoint(args.checkpoint)
    size = checkpoint['image_size']
    tensors = [torch.from_numpy(np.array(load_rgb(p, size), dtype=np.float32))[None] for p in args.images]
    with torch.inference_mode():
        reference = [model(t).numpy() for t in tensors]
    folders = create_output_folders(args.output, ['models', 'fixtures', 'metadata'])
    if args.format == 'tflite':
        # Official converter currently requires Linux; use Colab or a Linux venv.
        import litert_torch
        from ai_edge_litert.interpreter import Interpreter
        artifact = folders['models'] / 'retinascan.tflite'
        with torch.no_grad():
            edge = litert_torch.convert(model, (tensors[0],))
        edge.export(str(artifact))
        interpreter = Interpreter(model_path=str(artifact))
        interpreter.allocate_tensors()
        input_info, output_info = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        if input_info['shape'].tolist() != [1, size, size, 3] or input_info['dtype'] != np.float32:
            raise RuntimeError('Converter changed expected NHWC float32 contract')
        if output_info['shape'].tolist() != [1, 4] or output_info['dtype'] != np.float32:
            raise RuntimeError('Converter changed expected output contract')
        def predict(t):
            interpreter.set_tensor(input_info['index'], t.numpy())
            interpreter.invoke()
            return interpreter.get_tensor(output_info['index'])
    else:
        artifact = folders['models'] / 'retinascan.torchscript.pt'
        with torch.no_grad():
            traced = torch.jit.trace(model, tensors[0])
        torch.jit.save(traced, str(artifact))
        runtime = torch.jit.load(str(artifact)).eval()
        def predict(t):
            with torch.inference_mode():
                return runtime(t).numpy()
    max_error, durations = 0.0, []
    predict(tensors[0])  # warmup, excluded from timing
    for index, (tensor, expected) in enumerate(zip(tensors, reference)):
        start = time.perf_counter()
        actual = predict(tensor)
        durations.append((time.perf_counter() - start) * 1000)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        np.testing.assert_array_equal((actual > .5).sum(1), (expected > .5).sum(1))
        if np.any(np.diff(actual, axis=1) > 1e-6):
            raise RuntimeError('Exported cumulative probabilities are not ordered')
        max_error = max(max_error, float(np.max(np.abs(actual-expected))))
        np.save(folders['fixtures'] / f'fixture_{index}_rgb.npy', tensor.numpy())
        np.save(folders['fixtures'] / f'fixture_{index}_output.npy', expected)
    metadata = {
        'environment': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                        if d.metadata['Name'] in ['torch', 'torchvision', 'litert-torch', 'ai-edge-litert', 'numpy', 'Pillow']},
        'format_version': 2, 'architecture': ARCHITECTURE, 'preprocessing': PREPROCESSING,
        'model_file': artifact.relative_to(args.output).as_posix(), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'size_bytes': artifact.stat().st_size, 'labels': LABELS,
        'input': {'shape': [1, size, size, 3], 'dtype': 'float32', 'range': [0,255], 'channels': 'RGB',
                  'resize': 'EXIF orientation, RGB conversion, bilinear stretch to square; no crop or CLAHE',
                  'normalization': 'Inside model; do not normalize in app'},
        'output': {'shape': [1,4], 'meaning': 'P(grade > k), k=0..3',
                   'grade': 'count(probability > 0.5)',
                   'class_probabilities': '[1-p0, p0-p1, p1-p2, p2-p3, p3]'},
        'temperature': checkpoint['temperature'], 'dataset_sha256': checkpoint['dataset_sha256'],
        'parity': {'fixtures': len(tensors), 'max_absolute_error': max_error, 'grade_agreement': True},
        'host_runtime_mean_ms': float(np.mean(durations)), 'timing_note': 'Host only, not an Android benchmark',
    }
    # Written last: the bundle is only ready if all parity checks passed.
    (folders['metadata'] / 'model_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Validated export:', artifact)


if __name__ == '__main__':
    main()
