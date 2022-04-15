import init_path
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image
import argparse
from datasets.pascal_voc import PascalVOC


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--output',
                        help='output path',
                        required=True,
                        type=str)
    # parser.add_argument('--eval_set', type=str, default='val')
    parser.add_argument('--data_root', type=str, default='voc12/VOC+SBD/VOCdevkit/VOC2012')
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()
    return args


def run(output_dir, data_root, split='val'):
    dataset = VOCSemanticSegmentationDataset(split=split, data_dir=data_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    for id in dataset.ids:
        cls_labels = np.array(Image.open(os.path.join(output_dir, id + '.png')))
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
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

    fmt = ','.join([output_dir,
                    f'{np.nanmean(iou):.4f}', f'{np.mean(fp[1:]):.4f}', f'{np.mean(fn[1:]):.4f}'] + [f'{i:.4f}' for i in iou])
    print(fmt)
    with open(os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'record.csv'), mode='a') as fp:
        fp.write(fmt + '\n')

if __name__ == '__main__':
    args = parse_args()
    run(args.output, args.data_root, args.split)
