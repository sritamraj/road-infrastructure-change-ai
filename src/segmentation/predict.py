import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import albumentations as A
import numpy as np
import torch
from PIL import Image

from src.segmentation.models import build_model


def main():
    parser = argparse.ArgumentParser(description='Run U-Net road segmentation inference.')
    parser.add_argument('--image', default='data/raw/deepglobe/train/images/142436_sat.jpg')
    parser.add_argument('--output', default='outputs/predictions/142436_pred.png')
    parser.add_argument('--checkpoint', default='outputs/checkpoints/unet_best.pt')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    image = np.array(Image.open(image_path).convert('RGB'))
    original_size = (image.shape[1], image.shape[0])

    transform = A.Compose([
        A.Resize(height=args.size, width=args.size, interpolation=1),
        A.Normalize(),
    ])
    transformed = transform(image=image)
    tensor = torch.from_numpy(transformed['image'].transpose(2, 0, 1)).float().unsqueeze(0)

    model = build_model('unet')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].numpy()

    prediction = (probability >= args.threshold).astype(np.uint8) * 255
    prediction = np.array(Image.fromarray(prediction).resize(original_size, Image.Resampling.NEAREST))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction).save(output_path)

    print(f'Image: {image_path}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Output: {output_path}')
    print(f'Prediction road pixels: {int(np.count_nonzero(prediction))}')
    print('U-NET PREDICTION COMPLETE')


if __name__ == '__main__':
    main()
