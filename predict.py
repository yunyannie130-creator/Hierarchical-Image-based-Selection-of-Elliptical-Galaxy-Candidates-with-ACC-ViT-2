"""Run the released two-stage model on a directory of SDSS RGB cutouts."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from inference import UnlabeledImageDataset, predict_unlabeled_two_layer_model
from utils_p import val_get_transform

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--stage1', type=Path, default=Path('weights/stage1_accvit.pth'))
    parser.add_argument('--stage2', type=Path, default=Path('weights/stage2_accvit.pth'))
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    for checkpoint in (args.stage1, args.stage2):
        if not checkpoint.is_file():
            parser.error(f'Checkpoint not found: {checkpoint}. Download both weights from Releases.')
    dataset = UnlabeledImageDataset(args.images, val_get_transform())
    for path in dataset.image_paths:
        with Image.open(path) as im:
            if im.width != im.height:
                parser.error(f'Input must be square: {path} ({im.size})')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    predict_unlabeled_two_layer_model(loader, args.stage1, args.stage2,
                                      torch.device(args.device), output_csv=args.output)

if __name__ == '__main__':
    main()
