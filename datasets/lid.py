import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
import torchvision.transforms as T
import imageio
from utils import imutils


class LIDDataset(Dataset):
    """
    LID classification dataset Class for training phase.

    Examples:
    >>> dataset = LIDDataset(root='/home/deserts/WSSEG/data/LID/ILSVRC_DET')
    >>> print('#samples:', len(dataset))
    >>> print(dataset.class_weights)

    """
    CLASSES = [
        "background", "accordion", "airplane", "ant", "antelope", "apple", "armadillo", "artichoke", "axe", "baby_bed",
        "backpack", "bagel", "balance_beam", "banana", "band_aid", "banjo", "baseball", "basketball", "bathing_cap",
        "beaker", "bear", "bee", "bell_pepper", "bench", "bicycle", "binder", "bird", "bookshelf", "bow_tie", "bow",
        "bowl", "brassiere", "burrito", "bus", "butterfly", "camel", "can_opener", "car", "cart", "cattle", "cello",
        "centipede", "chain_saw", "chair", "chime", "cocktail_shaker", "coffee_maker", "computer_keyboard",
        "computer_mouse", "corkscrew", "cream", "croquet_ball", "crutch", "cucumber", "cup_or_mug", "diaper",
        "digital_clock", "dishwasher", "dog", "domestic_cat", "dragonfly", "drum", "dumbbell", "electric_fan",
        "elephant", "face_powder", "fig", "filing_cabinet", "flower_pot", "flute", "fox", "french_horn", "frog",
        "frying_pan", "giant_panda", "goldfish", "golf_ball", "golfcart", "guacamole", "guitar", "hair_dryer",
        "hair_spray", "hamburger", "hammer", "hamster", "harmonica", "harp", "hat_with_a_wide_brim", "head_cabbage",
        "helmet", "hippopotamus", "horizontal_bar", "horse", "hotdog", "iPod", "isopod", "jellyfish", "koala_bear",
        "ladle", "ladybug", "lamp", "laptop", "lemon", "lion", "lipstick", "lizard", "lobster", "maillot", "maraca",
        "microphone", "microwave", "milk_can", "miniskirt", "monkey", "motorcycle", "mushroom", "nail", "neck_brace",
        "oboe", "orange", "otter", "pencil_box", "pencil_sharpener", "perfume", "person", "piano", "pineapple",
        "ping-pong_ball", "pitcher", "pizza", "plastic_bag", "plate_rack", "pomegranate", "popsicle", "porcupine",
        "power_drill", "pretzel", "printer", "puck", "punching_bag", "purse", "rabbit", "racket", "ray", "red_panda",
        "refrigerator", "remote_control", "rubber_eraser", "rugby_ball", "ruler", "salt_or_pepper_shaker", "saxophone",
        "scorpion", "screwdriver", "seal", "sheep", "ski", "skunk", "snail", "snake", "snowmobile", "snowplow",
        "soap_dispenser", "soccer_ball", "sofa", "spatula", "squirrel", "starfish", "stethoscope", "stove", "strainer",
        "strawberry", "stretcher", "sunglasses", "swimming_trunks", "swine", "syringe", "table", "tape_player",
        "tennis_ball", "tick", "tie", "tiger", "toaster", "traffic_light", "train", "trombone", "trumpet", "turtle",
        "tv_or_monitor", "unicycle", "vacuum", "violin", "volleyball", "waffle_iron", "washer", "water_bottle",
        "watercraft", "whale", "wine_bottle", "zebra"
    ]

    CLASS_IDX = {
        "background": 0, "accordion": 1, "airplane": 2, "ant": 3, "antelope": 4, "apple": 5, "armadillo": 6,
        "artichoke": 7, "axe": 8, "baby_bed": 9, "backpack": 10, "bagel": 11, "balance_beam": 12, "banana": 13,
        "band_aid": 14, "banjo": 15, "baseball": 16, "basketball": 17, "bathing_cap": 18, "beaker": 19, "bear": 20,
        "bee": 21, "bell_pepper": 22, "bench": 23, "bicycle": 24, "binder": 25, "bird": 26, "bookshelf": 27,
        "bow_tie": 28, "bow": 29, "bowl": 30, "brassiere": 31, "burrito": 32, "bus": 33, "butterfly": 34, "camel": 35,
        "can_opener": 36, "car": 37, "cart": 38, "cattle": 39, "cello": 40, "centipede": 41, "chain_saw": 42,
        "chair": 43, "chime": 44, "cocktail_shaker": 45, "coffee_maker": 46, "computer_keyboard": 47,
        "computer_mouse": 48, "corkscrew": 49, "cream": 50, "croquet_ball": 51, "crutch": 52, "cucumber": 53,
        "cup_or_mug": 54, "diaper": 55, "digital_clock": 56, "dishwasher": 57, "dog": 58, "domestic_cat": 59,
        "dragonfly": 60, "drum": 61, "dumbbell": 62, "electric_fan": 63, "elephant": 64, "face_powder": 65, "fig": 66,
        "filing_cabinet": 67, "flower_pot": 68, "flute": 69, "fox": 70, "french_horn": 71, "frog": 72, "frying_pan": 73,
        "giant_panda": 74, "goldfish": 75, "golf_ball": 76, "golfcart": 77, "guacamole": 78, "guitar": 79,
        "hair_dryer": 80, "hair_spray": 81, "hamburger": 82, "hammer": 83, "hamster": 84, "harmonica": 85, "harp": 86,
        "hat_with_a_wide_brim": 87, "head_cabbage": 88, "helmet": 89, "hippopotamus": 90, "horizontal_bar": 91,
        "horse": 92, "hotdog": 93, "iPod": 94, "isopod": 95, "jellyfish": 96, "koala_bear": 97, "ladle": 98,
        "ladybug": 99, "lamp": 100, "laptop": 101, "lemon": 102, "lion": 103, "lipstick": 104, "lizard": 105,
        "lobster": 106, "maillot": 107, "maraca": 108, "microphone": 109, "microwave": 110, "milk_can": 111,
        "miniskirt": 112, "monkey": 113, "motorcycle": 114, "mushroom": 115, "nail": 116, "neck_brace": 117,
        "oboe": 118, "orange": 119, "otter": 120, "pencil_box": 121, "pencil_sharpener": 122, "perfume": 123,
        "person": 124, "piano": 125, "pineapple": 126, "ping-pong_ball": 127, "pitcher": 128, "pizza": 129,
        "plastic_bag": 130, "plate_rack": 131, "pomegranate": 132, "popsicle": 133, "porcupine": 134,
        "power_drill": 135, "pretzel": 136, "printer": 137, "puck": 138, "punching_bag": 139, "purse": 140,
        "rabbit": 141, "racket": 142, "ray": 143, "red_panda": 144, "refrigerator": 145, "remote_control": 146,
        "rubber_eraser": 147, "rugby_ball": 148, "ruler": 149, "salt_or_pepper_shaker": 150, "saxophone": 151,
        "scorpion": 152, "screwdriver": 153, "seal": 154, "sheep": 155, "ski": 156, "skunk": 157, "snail": 158,
        "snake": 159, "snowmobile": 160, "snowplow": 161, "soap_dispenser": 162, "soccer_ball": 163, "sofa": 164,
        "spatula": 165, "squirrel": 166, "starfish": 167, "stethoscope": 168, "stove": 169, "strainer": 170,
        "strawberry": 171, "stretcher": 172, "sunglasses": 173, "swimming_trunks": 174, "swine": 175, "syringe": 176,
        "table": 177, "tape_player": 178, "tennis_ball": 179, "tick": 180, "tie": 181, "tiger": 182, "toaster": 183,
        "traffic_light": 184, "train": 185, "trombone": 186, "trumpet": 187, "turtle": 188, "tv_or_monitor": 189,
        "unicycle": 190, "vacuum": 191, "violin": 192, "volleyball": 193, "waffle_iron": 194, "washer": 195,
        "water_bottle": 196, "watercraft": 197, "whale": 198, "wine_bottle": 199, "zebra": 200,
        'ambiguous': 255
    }

    NUM_CLASS = 201

    CLASS_WEIGHTS = [473.6381, 239.2325, 679.1953, 192.6128, 296.9130, 581.4164, 411.8366,
                     338.0693, 217.6101, 306.8587, 553.1591, 613.7173, 428.7614, 614.7218,
                     511.0136, 655.4887, 405.6102, 268.2821, 508.9363, 130.9150, 569.0833,
                     532.7589, 295.9771, 244.5280, 699.4320, 11.2084, 391.2448, 508.2476,
                     512.4079, 153.5548, 554.7932, 581.4164, 155.2687, 87.9408, 279.0453,
                     703.3614, 45.7596, 168.8067, 395.7798, 445.5457, 485.2649, 615.7295,
                     63.2421, 624.9501, 812.9762, 278.4248, 309.1317, 426.8125, 601.9151,
                     305.3618, 639.8552, 514.5137, 634.4510, 154.3118, 620.8182, 604.8229,
                     490.9739, 5.5632, 115.1073, 247.7540, 228.3252, 407.3698, 447.1369,
                     237.5680, 684.1439, 695.5463, 600.9520, 297.6189, 507.5608, 149.6394,
                     679.1953, 168.8826, 482.7699, 447.6698, 338.3739, 723.6898, 541.9841,
                     506.1927, 155.0124, 551.5345, 784.1232, 598.0812, 650.9445, 477.2490,
                     625.9917, 386.4146, 210.1819, 534.2745, 139.7303, 471.8530, 732.1540,
                     241.3850, 594.2959, 524.5740, 742.2826, 367.1505, 367.8697, 629.1374,
                     374.0986, 171.9757, 258.3184, 490.9739, 446.0748, 710.0095, 64.6574,
                     209.9469, 627.0367, 677.9693, 151.5107, 420.5991, 648.6960, 363.5963,
                     48.7279, 188.5517, 490.9739, 592.4211, 545.1306, 602.8812, 378.2427,
                     443.9657, 611.7182, 639.8552, 669.5098, 12.6459, 225.5826, 610.7236,
                     667.1314, 344.8990, 449.8144, 569.9469, 593.3570, 409.1449, 551.5345,
                     328.3173, 555.6139, 575.1838, 588.7069, 797.4416, 566.5083, 318.3008,
                     181.8862, 656.6346, 242.3194, 385.6211, 421.5432, 532.7589, 603.8505,
                     459.7246, 642.0427, 559.7541, 428.7614, 456.3730, 643.1421, 270.4068,
                     269.6303, 642.0427, 421.0706, 448.7395, 43.7145, 614.7218, 506.1927,
                     638.7670, 446.0748, 263.7605, 567.3640, 412.2887, 394.1186, 418.7235,
                     215.1174, 518.0621, 479.6871, 722.2981, 187.4227, 498.7981, 196.2356,
                     426.3280, 55.9587, 332.3850, 579.6219, 591.4882, 340.2129, 417.7920,
                     572.5534, 450.3537, 312.4750, 569.9469, 465.4213, 132.7660, 171.7398,
                     736.4608, 640.9471, 370.4093, 567.3640, 570.8131, 524.5740, 482.1502,
                     45.2470, 296.4444, 455.2667, 448.7395]

    BALANCED_CLASS_WEIGHTS = [1.1364, 1.0078, 0.9619, 1.0036, 0.8682, 1.0450, 1.0217, 1.0433, 1.0349,
                              0.9726, 0.9806, 1.0594, 0.9963, 1.0778, 1.1625, 1.1465, 1.0760, 0.9989,
                              0.9907, 0.9245, 1.0020, 0.9963, 0.9984, 1.0790, 1.0411, 0.9658, 0.9761,
                              1.0882, 1.0479, 0.7027, 1.1618, 1.0255, 1.0277, 1.0083, 1.0647, 0.9673,
                              0.7352, 1.0094, 1.0277, 1.0712, 1.0126, 1.1264, 0.4560, 1.0110, 1.0073,
                              0.8837, 0.8961, 0.9668, 0.9968, 0.9746, 1.1231, 1.1583, 1.0405, 0.7644,
                              1.1108, 1.0399, 0.9629, 0.9188, 0.9179, 1.0244, 1.0089, 1.1114, 1.0748,
                              0.9927, 0.9999, 1.0046, 1.0571, 0.9756, 1.0950, 0.9791, 1.0876, 1.0456,
                              0.9501, 1.0041, 1.0473, 1.0305, 1.0623, 0.9968, 0.8339, 1.0766, 1.0299,
                              1.0179, 0.9836, 0.9989, 1.0919, 1.1121, 0.9477, 0.9816, 0.8339, 1.0283,
                              1.1486, 1.0559, 0.9761, 0.9922, 1.0179, 1.0041, 1.0288, 1.0742, 1.0078,
                              0.8118, 0.9079, 0.9871, 1.0467, 1.0857, 1.0223, 1.0089, 1.0888, 0.9989,
                              0.7009, 0.9157, 1.0062, 1.0582, 0.9948, 1.0057, 0.9771, 0.9846, 1.0623,
                              1.0612, 0.8903, 1.0115, 0.9417, 0.9624, 0.9600, 0.0641, 0.9771, 1.0201,
                              1.0712, 0.9412, 1.0073, 1.0676, 1.0147, 1.0052, 0.9968, 0.9937, 1.1063,
                              0.9741, 1.0152, 1.1140, 1.0577, 0.9591, 0.9707, 1.0559, 1.0217, 0.9907,
                              0.9576, 1.0305, 0.9968, 1.0606, 0.9826, 1.0261, 1.0676, 0.9932, 0.9968,
                              0.9459, 1.0104, 1.1717, 1.0005, 1.0294, 1.0206, 1.1418, 1.0808, 0.9756,
                              1.1681, 0.9999, 1.0456, 0.9796, 1.0099, 1.0888, 0.7557, 1.0094, 0.9984,
                              1.0944, 0.8903, 1.0730, 0.9948, 1.1198, 0.3847, 1.0026, 0.9751, 1.0158,
                              1.0496, 1.0294, 0.9856, 1.0190, 1.0089, 1.1000, 1.0416, 0.9821, 0.7187,
                              1.0845, 1.0659, 1.0479, 1.0671, 1.0288, 1.0217, 1.0223, 0.9731, 0.9781,
                              1.0577, 1.0456]

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, split='train', root='./data', crop_size=512, scale=(0.5, 1.0), clutter_penalty=1):
        super().__init__()
        self.root = root
        assert split == 'train', 'Only support training set.'
        self.clutter_penalty = clutter_penalty

        self.transform = T.Compose([T.RandomResizedCrop(crop_size, scale=scale),
                                    T.RandomHorizontalFlip(),
                                    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)],
                                                  p=1.0),
                                    T.ToTensor(),
                                    T.Normalize(self.MEAN, self.STD)])

        anno_path = os.path.join(self.root, 'Annotations', 'cls_labels.npy')
        cls_labels_anno = np.load(anno_path, allow_pickle=True).item()
        self.image_paths = [p+'.JPEG' for p in cls_labels_anno['files']]
        self.labels = cls_labels_anno['labels']

        assert len(self.image_paths) == self.labels.shape[0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, 'Data', self.image_paths[idx])
        image = Image.open(image_path)
        image = image.convert('RGB')
        label = self.labels[idx]
        image = self.transform(image)
        return image, label, image_path

    @property
    def class_weights(self):
        class_sample_counts = self.labels.sum(0)
        assert class_sample_counts.shape[0] == self.NUM_CLASS - 1
        return torch.from_numpy(class_sample_counts.sum() / (class_sample_counts + 1))

    @property
    def sample_weights(self):
        class_weights = self.class_weights
        # (N, C) * (1, C) -> (N, C)
        labels = torch.from_numpy(self.labels)
        weights = labels.type_as(class_weights) * class_weights[None,]
        weights = weights.sum(dim=-1).type(torch.double)
        if self.clutter_penalty > 1:
            count = labels[:, :self.CLASS_IDX['person']].sum(-1) + labels[:, self.CLASS_IDX['person']+1:].sum(-1)
            count[count < 1] = np.inf
            p = 1 / (count ** self.clutter_penalty + 1)
        elif self.clutter_penalty == 1:
            p = 1 / (labels.sum(-1) + 1)
        else:
            p = 1.0
        weights = weights * p
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


class LIDDatasetSegMS(LIDDataset):

    def __init__(self, split='val', root='./data', scales=(1.0,)):
        self.scales = scales
        self.split = split
        self.root = root

        # train/val/test splits are pre-cut
        if self.split == 'val':
            _split_f = os.path.join(self.root, 'val.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.invalid_indices = [12, 71, 99, 100, 123, 210, 215, 274, 294, 325, 333, 376, 416, 452, 472, 479, 491, 501, 519, 607, 639, 641, 649, 673, 748, 785, 797, 834, 871, 918, 946, 1061, 1100, 1125, 1132, 1133, 1140, 1146, 1153, 1208, 1215, 1255, 1361, 1490, 1572, 1597, 1651, 1667, 1680, 1695, 1745, 1769, 1774, 1802, 1839, 1872, 1903, 1907, 1940, 1992, 2067, 2165, 2173, 2191, 2291, 2334, 2335, 2336, 2353, 2362, 2381, 2404, 2406, 2413, 2416, 2435, 2468, 2518, 2540, 2586, 2655, 2663, 2679, 2708, 2778, 2787, 2811, 2821, 2861, 2913, 2929, 2944, 2959, 2960, 2962, 3326, 3342, 3343, 3377, 3409, 3410, 3411, 3412, 3435, 3475, 3496, 3543, 3581, 3594, 3603, 3659, 3679, 3773, 3802, 3817, 3819, 3824, 3907, 3971, 3980, 3991, 4062, 4077, 4127, 4160, 4195, 4211, 4285, 4326, 4349, 4392, 4480, 4492, 4496, 4537, 4538, 4577, 4579, 4597, 4600, 4614]

        self.names = []
        self.image_paths = []
        self.mask_paths = []
        with open(_split_f, "r") as lines:
            for idx, line in enumerate(lines):
                # if idx in invalid_indices:
                #     continue
                _name = line.strip()
                self.names.append(_name)
                _image = os.path.join(self.root, 'LID_track1', split, _name + '.JPEG')
                assert os.path.isfile(_image), '%s not found' % _image
                self.image_paths.append(_image)

                if self.split != 'test':
                    _mask = os.path.join(self.root, 'track1_val_annotations_raw', _name + '.png')
                    assert os.path.isfile(_mask), '%s not found' % _mask
                    self.mask_paths.append(_mask)

        if self.split != 'test':
            assert (len(self.image_paths) == len(self.mask_paths))

        self.transform = TorchvisionNormalize()

    def __getitem__(self, idx):
        name = self.names[idx]
        # img = imageio.imread(self.image_paths[idx])
        img = Image.open(self.image_paths[idx])
        img = img.convert('RGB')
        img = np.array(img)
        ms_img_list = []

        if self.split != 'test':
            mask = Image.open(self.mask_paths[idx])
            mask = np.array(mask)
            unique_labels = np.unique(mask)

            # ambigious
            if unique_labels[-1] == self.CLASS_IDX['ambiguous']:
                unique_labels = unique_labels[:-1]

            # ignoring BG
            labels = torch.zeros(self.NUM_CLASS - 1)
            if unique_labels[0] == self.CLASS_IDX['background']:
                unique_labels = unique_labels[1:]
            unique_labels -= 1  # shifting since no BG class

            # assert unique_labels.size > 0, 'No labels found in %s' % self.mask_paths[idx]
            # labels[unique_labels.tolist()] = 1
            if unique_labels.size > 0:
                labels[unique_labels.tolist()] = 1
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            labels = np.ones(self.NUM_CLASS - 1)

        # if img.shape[0] > 1000 and img.shape[1] > 1000:
        #     raise ValueError
        # if idx in self.invalid_indices:
        if img.shape[0] > 1000 or img.shape[1] > 1000:
            # 将尺寸过大的图片缩小后再处理，注意最终结果是被缩小了的，要专门放大回去
            # print(name, img.shape[0], img.shape[1])
            scales = (1, 0.5, 0.75)
            while img.shape[0] > 1000 or img.shape[1] > 1000:
                img = imutils.pil_rescale(img, 0.5, order=3)
                mask = imutils.pil_rescale(mask, 0.5, order=0)
        else:
            scales = self.scales

        for s in scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.transform(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        # if len(self.scales) == 1:
        #     ms_img_list = ms_img_list[0]

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": labels, "mask": mask.copy()}
        return out


class LIDDatasetSegMSTrainset(LIDDataset):
    """
    For pseudo label generation
    """
    def __init__(self, split='train', root='./data', scales=(1.0,)):
        self.root = root
        self.scales = scales
        assert split == 'train', 'Only support training set.'

        self.transform = TorchvisionNormalize()

        anno_path = os.path.join(self.root, 'Annotations', 'cls_labels.npy')
        cls_labels_anno = np.load(anno_path, allow_pickle=True).item()
        self.image_paths = [os.path.join(self.root, 'Data', p+'.JPEG') for p in cls_labels_anno['files']]
        self.names = [p for p in cls_labels_anno['files']]
        self.labels = cls_labels_anno['labels']

        assert len(self.image_paths) == self.labels.shape[0]

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(self.image_paths[idx])
        img = img.convert('RGB')
        img = np.array(img)

        if img.shape[0] > 1000 or img.shape[1] > 1000:
            # 将尺寸过大的图片缩小后再处理，注意最终结果是被缩小了的，要专门放大回去
            # print(name, img.shape[0], img.shape[1])
            scales = (1.0, 0.5, 0.75)
            while img.shape[0] > 1000 or img.shape[1] > 1000:
                img = imutils.pil_rescale(img, 0.5, order=3)
        else:
            scales = self.scales

        ms_img_list = []

        labels = self.labels[idx]

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
               "label": labels, "mask": torch.zeros(img.shape[0], img.shape[1], dtype=torch.int)}
        return out


if __name__ == '__main__':
    import doctest
    doctest.testmod()