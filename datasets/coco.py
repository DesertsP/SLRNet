import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
import torchvision.transforms as T
from utils import imutils
import logging

try:
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO
except:
    logging.info('Pycocotools not installed!')


class COCOClassification(Dataset):
    """
    LID classification dataset Class for training phase.

    Examples:
    >>> dataset = COCOClassification(root='/home/deserts/WSSEG/data/coco', split='train')
    # >>> print('#samples:', len(dataset))
    # >>> dataset.class_weights
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASS = 81
    CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    CAT_ID_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14,
                  16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26,
                  31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38,
                  43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
                  56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62,
                  72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74,
                  85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

    # 使用sample weights重采样后的类别权重，未包含background
    BALANCED_CLASS_WEIGHTS = [0.0580, 0.9439, 0.3090, 1.1062, 1.1484, 1.0208, 0.9967, 0.6532, 1.0895,
                              0.9073, 1.3032, 1.3549, 1.7261, 0.6848, 0.9237, 0.7763, 0.7619, 1.1403,
                              1.1233, 1.1372, 1.1843, 1.1465, 1.0259, 1.0586, 0.6450, 0.9943, 0.5161,
                              1.0647, 1.2419, 1.4560, 1.3231, 1.4945, 0.8353, 1.4468, 1.2007, 1.1670,
                              1.3137, 1.3341, 1.2179, 0.3373, 1.1479, 0.3139, 0.7700, 0.6261, 0.7421,
                              0.3778, 0.9794, 1.0642, 1.0816, 1.0154, 1.0580, 1.1336, 1.5824, 1.0513,
                              1.4388, 0.9719, 0.2687, 0.7155, 0.6840, 0.8768, 0.2607, 0.8369, 0.6103,
                              0.7912, 1.1343, 1.0577, 0.9783, 0.7671, 1.1942, 0.7544, 2.6064, 0.4852,
                              0.9054, 0.5438, 0.7720, 0.8335, 1.4894, 1.1677, 2.2971, 1.4464]

    def __init__(self, split='train', root='./data', crop_size=512, scale=(0.5, 1.0)):
        super().__init__()
        self.root = root

        self.transform = T.Compose([T.RandomResizedCrop(crop_size, scale=scale),
                                    T.RandomHorizontalFlip(),
                                    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)],
                                                  p=1.0),
                                    T.ToTensor(),
                                    T.Normalize(self.MEAN, self.STD)])
        if split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'annotations/instances_train2014.json')
            self.root = os.path.join(root, 'train2014')
        else:
            print('val set')
            assert split == 'val'
            ann_file = os.path.join(root, 'annotations/instances_val2014.json')
            self.root = os.path.join(root, 'val2014')

        self.coco = COCO(ann_file)
        self.img_ids = sorted(list(self.coco.imgs.keys()))

        self.cls_labels = np.zeros((len(self), self.NUM_CLASS-1))
        for idx, iid in enumerate(self.img_ids):
            self.cls_labels[idx, :] = self._gen_cls_labels(iid)[1:]

    def __len__(self):
        return len(self.img_ids)

    def _gen_cls_labels(self, img_id):
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        cls_labels = np.zeros((self.NUM_CLASS,), dtype=np.float)
        for instance in annotations:
            cat = instance['category_id']
            if cat in self.CAT_ID_MAP:
                c = self.CAT_ID_MAP[cat]
                cls_labels[c] = 1
        return cls_labels

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        name = img_metadata['file_name']
        image = Image.open(os.path.join(self.root, name)).convert('RGB')

        label = self._gen_cls_labels(img_id)[1:]
        label = torch.from_numpy(label).float()
        # label = self.cls_labels[idx]   # discard the bg category
        image = self.transform(image)
        return image, label, name

    @property
    def class_weights(self):
        class_sample_counts = self.cls_labels.sum(0)
        assert class_sample_counts.shape[0] == self.NUM_CLASS - 1
        return torch.from_numpy(class_sample_counts.sum() / (class_sample_counts + 1))

    @property
    def sample_weights(self):
        class_weights = self.class_weights
        # (N, C) * (1, C) -> (N, C)
        labels = torch.from_numpy(self.cls_labels)
        weights = labels.type_as(class_weights) * class_weights[None,]
        weights = weights.sum(dim=-1).type(torch.double)
        weights = weights / (1+labels.sum(-1))
        return weights



class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class COCOSegmentation(COCOClassification):
    def __init__(self, split='val', root='./data'):
        super().__init__(split, root)
        # self.transform = TorchvisionNormalize()

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.img_ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
        # img = self.transform(img)
        return img, mask

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_ID_MAP:
                c = self.CAT_ID_MAP[cat]
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    # def _preprocess(self, ids, ids_file):
    #     print("Preprocessing mask, this will take a while." + \
    #         "But don't worry, it only run once for each split.")
    #     tbar = trange(len(ids))
    #     new_ids = []
    #     for i in tbar:
    #         img_id = ids[i]
    #         cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
    #         img_metadata = self.coco.loadImgs(img_id)[0]
    #         mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
    #                                   img_metadata['width'])
    #         # more than 1k pixels
    #         if (mask > 0).sum() > 1000:
    #             new_ids.append(img_id)
    #         tbar.set_description('Doing: {}/{}, got {} qualified images'.\
    #             format(i, len(ids), len(new_ids)))
    #     print('Found number of qualified images: ', len(new_ids))
    #     torch.save(new_ids, ids_file)
    #     return new_ids

class COCOSegmentationMS(COCOSegmentation):
    def __init__(self, split='train', root='./data', scales=(1.0,)):
        super().__init__(split, root)
        self.transform = TorchvisionNormalize()
        self.scales = scales

    def __getitem__(self, idx):
        img, mask = super(COCOSegmentationMS, self).__getitem__(idx)
        img = np.array(img)
        img_id = self.img_ids[idx]
        labels = self.cls_labels[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        name = img_metadata['file_name']

        ms_img_list = []
        scales = self.scales
        for s in scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.transform(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))

        size = (img.shape[0], img.shape[1])

        out = {"name": name, "img": ms_img_list, "size": size,
               "label": labels, "mask": mask}
        return out


if __name__ == '__main__':
    import doctest
    doctest.testmod()