import numpy as np
import os
# from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image
import argparse
import init_path
from datasets.pascal_voc import PascalVOC


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--output',
                        help='output path',
                        required=True,
                        type=str)
    parser.add_argument('--data_root', type=str, default='data/COCO/')
    args = parser.parse_args()
    return args


def run(output_dir, gt_dir):
    def pred_gen():
        # for idx in range(len(dataset)):
        for name in sorted(os.listdir(output_dir)):
            if name.startswith('.') or not name.endswith('.png'):
                continue
            pth = os.path.join(output_dir, name)
            pred = Image.open(pth)
            pred = np.array(pred).astype(np.uint8)
            yield pred.copy()

    def gt_gen():
        for name in sorted(os.listdir(output_dir)):
            if name.startswith('.') or not name.endswith('.png'):
                continue
            pth = os.path.join(gt_dir, name)
            mask = np.array(Image.open(pth)).astype(np.uint8)
            yield mask.copy()

    labels = gt_gen()
    preds = pred_gen()

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj #+ 1e-8
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    cls_names = PascalVOC.CLASSES[:21]
    print(output_dir)
    print(f'fp for BG: {fp[0]:.4f}, fn for BG: {fn[0]:.4f}')
    print(f'mean fp for FG: {np.mean(fp[1:]):.4f}, mean fn for FG: {np.mean(fn[1:]):.4f}')
    print(f'miou: {np.nanmean(iou)}')
    print()
    print(f'{"class": <15}\t {"iou":<6}\t {"fp":<6}\t {"fn":<6}')
    for name, i, j, k in zip(cls_names, iou, fp, fn):
        print(f'{name: <15}\t {i:.4f}\t {j:.4f}\t {k:.4f}')
    print()


if __name__ == '__main__':
    args = parse_args()
    run(args.output, args.data_root)
