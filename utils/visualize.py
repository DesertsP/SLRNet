import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils


def colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'uint8'
    cmap = []
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap.append((r, g, b))

    return cmap


""" 
Python implementation of the color map function for the PASCAL VOC data set. 
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


class Colorize(object):

    def __init__(self, n=22):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
    'tv/monitor'
]


def mask_rgb(masks, image_norm, n=22):
    # visualising masks
    masks_conf, masks_idx = torch.max(masks, 1)
    masks_conf = masks_conf - F.relu(masks_conf - 1, 0)

    masks_idx_rgb = _apply_cmap(masks_idx.cpu(), masks_conf.cpu(), n=n)
    return 0.3 * image_norm + 0.7 * masks_idx_rgb


def _apply_cmap(mask_idx, mask_conf, n=22):
    masks = []
    col = Colorize(n=n)
    mask_conf = mask_conf.float() / 255.0
    for mask, conf in zip(mask_idx.split(1), mask_conf.split(1)):
        m = col(mask).float()
        m = m * conf
        masks.append(m[None, ...])
    return torch.cat(masks, 0)


def make_grid(x_all, labels, class_names=None):
    # adding the labels to images
    bs, ch, h, w = x_all.size()
    x_all_new = torch.zeros(bs, ch, h + 16, w)
    _, y_labels_idx = torch.max(labels, -1)
    # class_names_offset = len(CLASSES) - labels.size(1)
    if class_names:
        class_names = class_names[1:]
    else:
        class_names = CLASSES[1:]
    for b in range(bs):
        label_idx = labels[b]
        label_names = [name for i, name in enumerate(class_names) if label_idx[i].item()]
        label_name = ", ".join(label_names)

        ndarr = x_all[b].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        arr = np.zeros((16, w, ch), dtype=ndarr.dtype)
        ndarr = np.concatenate((arr, ndarr), 0)
        im = Image.fromarray(ndarr)
        draw = ImageDraw.Draw(im)

        try:
            font = ImageFont.truetype("utils/fonts/UbuntuMono-R.ttf", 12)
        except ImportError:
            font = None

        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((5, 1), label_name, (255, 255, 255), font=font)
        im_np = np.array(im).astype(np.float)
        x_all_new[b] = (torch.from_numpy(im_np) / 255.0).permute(2, 0, 1)

    summary_grid = vutils.make_grid(x_all_new, nrow=1, padding=8, pad_value=0.9)
    return summary_grid
