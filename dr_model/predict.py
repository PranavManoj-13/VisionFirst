"""Run the same inference contract used by the Android export."""
import argparse
import numpy as np
import torch
from dr_model.data import load_rgb
from dr_model.model import LABELS, class_probabilities
from dr_model.export import load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)
    image = torch.from_numpy(np.array(load_rgb(args.image, checkpoint['image_size']), dtype=np.float32))[None]
    with torch.inference_mode():
        cumulative = model(image)
    grade = int((cumulative > .5).sum())
    print({'grade': grade, 'label': LABELS[grade], 'class_probabilities': class_probabilities(cumulative)[0].tolist()})
