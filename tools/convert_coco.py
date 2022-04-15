import init_path

from datasets.coco import COCOSegmentationMS
from datasets.utils import labelcolormap
from PIL import Image
import os
import tqdm

root = '../datasets/COCO'
if not os.path.exists(os.path.join(root, 'val_masks')):
    os.mkdir(os.path.join(root, 'val_masks'))
coco = COCOSegmentationMS(root=root, split='val')
for i in tqdm.tqdm(range(len(coco))):
    data = coco[i]
    Image.fromarray(data['mask']).save(os.path.join(root, 'val_masks', data['name'] + '.png'))
