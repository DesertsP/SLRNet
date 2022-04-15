import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
import datasets.transforms as tf
import torchvision.transforms as transforms
import random

from datasets.pascal_voc import VOCSegmentation


class VOCSegmentationSemi(VOCSegmentation):
    def __init__(self, *args, oversample_rate: int = 5, **kwargs):
        super(VOCSegmentationSemi, self).__init__(*args, **kwargs)
        if self.split == 'train' and oversample_rate > 1:
            self.images += self.images[-1464:] * (oversample_rate - 1)
            self.masks += self.masks[-1464:] * (oversample_rate - 1)

        # self.transform.segtransform = self.transform.segtransform[:-1]
        self.num_fully_sup = 1464 * oversample_rate

    def __getitem__(self, index):
        assert index >= 0, 'negative index not supported.'
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == self.CLASS_IDX['ambiguous']:
            unique_labels = unique_labels[:-1]

        # ignoring BG
        labels = torch.zeros(self.NUM_CLASS - 1)
        if unique_labels[0] == self.CLASS_IDX['background']:
            unique_labels = unique_labels[1:]
        unique_labels -= 1  # shifting since no BG class

        assert unique_labels.size > 0, 'No labels found in %s' % self.masks[index]
        labels[unique_labels.tolist()] = 1

        image, mask = self.transform(image, mask)
        if index >= len(self) - self.num_fully_sup:
            # fully-sup samples
            # print('full', index)
            return image, mask, labels, os.path.basename(self.images[index]), True
        else:
            # provide a `placeholder`  for weakly-sup samples
            # print('weak', index)
            return image, torch.zeros_like(mask), labels, os.path.basename(self.images[index]), False


if __name__ == '__main__':
    d = VOCSegmentationSemi(split='train', root='/home/deserts/VOC+SBD', crop_size=320, scale=(0.9, 1.0))
    print(len(d))
    d[9119]