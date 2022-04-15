from .pascal_voc import VOCSegmentation
from .pascal_voc_semi import VOCSegmentationSemi
from .lid import LIDDataset
from .coco import COCOClassification

DATASETS = {
    'voc': VOCSegmentation,
    'voc_semi': VOCSegmentationSemi,
    'lid': LIDDataset,
    'coco': COCOClassification
}

def get_dataset(name, *args, **kwargs):
    assert name in DATASETS, 'Dataset is not supported'
    return DATASETS[name](*args, **kwargs)
