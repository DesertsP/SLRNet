import init_path
import os
import numpy as np
import torch
from torch import multiprocessing
import torch.nn.functional as F
from utils.dcrf import crf_inference
from utils import config_parser
import argparse
from torch.utils.data import DataLoader, Subset
from models import get_model
from datasets.coco import COCOSegmentationMS
from PIL import Image, ImagePalette
from utils.utils import denorm
import imageio
from utils.meters import Progbar
import eval_coco
from datasets.utils import colormap

torch.multiprocessing.set_sharing_strategy('file_system')

TOPK = 8


def load_model_weights(model, weight_path):
    assert os.path.isfile(weight_path), 'resume checkpoint not found.'
    checkpoint = torch.load(weight_path, map_location={'cuda:0': 'cpu'})
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    # model.load_state_dict(
        # {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
    # model.load_state_dict()
    return model


def _get_palette():
    cmap = colormap()
    palette = ImagePalette.ImagePalette()
    for rgb in cmap:
        palette.getcolor(rgb)
    return palette


class ResultWriter:
    def __init__(self, palette, output_path, num_classes=21, verbose=False):
        self.palette = palette
        self.root = output_path
        self.verbose = verbose
        self.num_classes = num_classes

    def _mask_overlay(self, mask, image, alpha=0.3):
        """Creates an overlayed mask visualisation"""
        mask_rgb = self.__mask2rgb(mask)
        return alpha * image + (1 - alpha) * mask_rgb

    def __mask2rgb(self, mask):
        im = Image.fromarray(mask).convert("P")
        im.putpalette(self.palette)
        mask_rgb = np.array(im.convert("RGB"), dtype=np.float)
        return mask_rgb / 255.

    def save(self, img_name, img_orig, masks, gt_mask=None):
        # converting original image to [0, 255]
        img_orig255 = np.round(255. * img_orig).astype(np.uint8)
        img_orig255 = np.transpose(img_orig255, [1, 2, 0])
        img_orig255 = np.ascontiguousarray(img_orig255)

        pred = np.argmax(masks, 0)

        # CRF
        # pred_crf = crf_inference(img_orig255, masks, t=10, scale_factor=1, labels=21)
        pred_crf = crf_inference(img_orig255, masks, t=10, scale_factor=1, labels=self.num_classes)
        pred_crf = np.argmax(pred_crf, 0)

        filepath = os.path.join(self.root, 'pred', img_name + '.png')
        imageio.imsave(filepath, pred.astype(np.uint8))

        filepath = os.path.join(self.root, "crf", img_name + '.png')
        imageio.imsave(filepath, pred_crf.astype(np.uint8))

        if self.verbose:
            mask_gt = gt_mask.numpy()
            masks_all = np.concatenate([pred, pred_crf, mask_gt], 1).astype(np.uint8)
            images = np.concatenate([img_orig] * 3, 2)
            images = np.transpose(images, [1, 2, 0])

            overlay = self._mask_overlay(masks_all, images)
            overlay255 = np.round(overlay * 255.).astype(np.uint8)

            filepath = os.path.join(self.root, "../vis", img_name + '.png')
            imageio.imsave(filepath, overlay255)


def _worker(process_id, model, datasets, scales, output_path, num_workers=2, verbose=1, bg_pow=3, fp_cut=0.1, use_cls_label=True, co_clustering=True):
    databin = datasets[process_id]
    n_gpus = torch.cuda.device_count()
    # num_workers 设置为0来降低推理速度，给writer进程（主要是CRF）留足够的时间来处理，防止任务队列爆炸导致内存不足。
    data_loader = DataLoader(databin, shuffle=False, num_workers=1, pin_memory=False)
    writer = ResultWriter(palette=_get_palette(), output_path=output_path, num_classes=81, verbose=True)
    pool = multiprocessing.Pool(processes=num_workers)
    progbar = Progbar(len(data_loader), prefix="infer", verbose=verbose)
    with torch.no_grad(), torch.cuda.device(process_id):
        model.cuda()
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            mask = pack['mask'][0]
            size = pack['size']

            # 断点
            img_name = os.path.basename(img_name)
            if os.path.isfile(os.path.join(output_path, "crf", img_name + '.png')) and os.path.getsize(os.path.join(output_path, "crf", img_name + '.png')) > 0:
                continue
            #
            # for each scales
            out_cam_list = []
            cls_list = []
            for img in pack['img']:
                # pack: img list, batch_size=1, 一个尺度是打包一起（flip），先for each scale，再去掉batch轴，得到(2,3,h,w)
                cls, out_cam = model(img[0].cuda(non_blocking=True), single_scale=not co_clustering)
                out_cam = out_cam[0] + out_cam[1].flip(-1)
                cls = (cls[0] + cls[1]) / 2.0
                cls_list.append(cls)
                # note that the batch axis was eliminated, add it again for interpolate
                out_cam = torch.unsqueeze(out_cam, 0)
                out_cam = F.interpolate(out_cam, size, mode='bilinear', align_corners=True)
                # remove batch axis again
                out_cam_list.append(out_cam[0])
            masks_pred = torch.sum(torch.stack(out_cam_list, 0), 0) / (2*len(out_cam_list))
            cls_pred = torch.sum(torch.stack(cls_list, 0), 0) / len(cls_list)
            # valid category idx
            if not use_cls_label:
                cls_sigmoid = torch.sigmoid(cls_pred)
                # cls_sigmoid = cls_sigmoid.mean(0)
                cls_label = (cls_sigmoid > fp_cut) * (cls_sigmoid >= torch.topk(cls_sigmoid, k=TOPK)[0][-1])
            else:
                cls_label = label

            image_with_origin_size = pack['img'][np.where(np.array(scales) == 1)[0][0]][0][0]
            image = denorm(image_with_origin_size).cpu().numpy()
            masks_pred = masks_pred.cpu()
            cls_label = cls_label.type_as(masks_pred)
            masks_pred[1:, :, :] *= cls_label.view(cls_label.size(0), 1, 1).cpu()
            masks_pred[0, :, :] = torch.pow(masks_pred[0], bg_pow)
            # TODO: "person" penalty
            masks_pred[1, :, :] = torch.pow(masks_pred[1], bg_pow*2)

            masks_pred = masks_pred.numpy()

            if num_workers > 0:
                result = pool.apply_async(writer.save, args=(img_name, image, masks_pred, mask))
                # inference 200 次以后，停下来等子线程，防止队列太大导致的内存爆满
                if iter % 200 == process_id * 50:
                    result.get()
            else:
                writer.save(img_name, image, masks_pred, mask)

            if process_id == n_gpus - 1:
                progbar.update(iter + 1)
    pool.close()
    pool.join()

def split_dataset(dataset, n_splits):
    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--output',
                        help='output path',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                        help='checkpoint path',
                        required=True,
                        type=str)
    parser.add_argument('--data_root', type=str, default='voc12/VOC+SBD/VOCdevkit/VOC2012')
    parser.add_argument('--bg_pow', type=float, default=1)
    parser.add_argument('--use_cls_label', type=int, default=1)
    parser.add_argument('--fp_cut', type=float, default=0.3)
    parser.add_argument('--cluster_iters', type=int, default=1)
    parser.add_argument('--co_cluster', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='val')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    VERBOSE = 1
    NUM_WORKERS=4
    print(vars(args))
    config = config_parser.parse_config(args.config, args.opts)
    print(config)
    if args.dataset == 'val':
        # ground-truth provided
        test_mode = False
        dataset = COCOSegmentationMS(split='val', root=args.data_root, scales=config.test.scales)
    elif args.dataset == 'minival':
        test_mode = False
        dataset = COCOSegmentationMS(split='val', root=args.data_root, scales=config.test.scales)
        dataset = Subset(dataset, np.arange(0, len(dataset), 10))
    else:
        test_mode = True
        raise NotImplementedError
        # dataset = LIDDatasetSegMSTrainset(split='train', root=args.data_root, scales=config.test.scales)
        # if args.subset >= 0:
        #     dataset = Subset(dataset, np.arange(args.subset, len(dataset), 8))
    n_gpus = torch.cuda.device_count()
    datasets = split_dataset(dataset, n_gpus)


    model = get_model(**config.model)
    model = load_model_weights(model, args.checkpoint)
    model.eval()
    # modify clustering iterations
    if hasattr(model, 'factorization_reconstruction') and hasattr(model.factorization_reconstruction, 'num_iters'):
        print('clustering iters:', args.cluster_iters)
        model.factorization_reconstruction.num_iters = args.cluster_iters

    if not os.path.exists(os.path.join(args.output, 'pred')):
        os.makedirs(os.path.join(args.output, 'pred'))
    if not os.path.exists(os.path.join(args.output, 'crf')):
        os.makedirs(os.path.join(args.output, 'crf'))
    if not os.path.exists(os.path.join(args.output, '../vis')):
        os.makedirs(os.path.join(args.output, '../vis'))

    if n_gpus == 1:
        _worker(0, model, datasets, config.test.scales, args.output, num_workers=2,
                bg_pow=args.bg_pow, fp_cut=args.fp_cut, use_cls_label=args.use_cls_label, co_clustering=args.co_cluster)
    else:
        multiprocessing.spawn(_worker, nprocs=n_gpus,
                              args=(model, datasets, config.test.scales, args.output, NUM_WORKERS, VERBOSE, args.bg_pow, args.fp_cut, args.use_cls_label, args.co_cluster),
                              join=True)

    torch.cuda.empty_cache()
    if not test_mode:
        eval_coco.run(os.path.join(args.output, 'pred'), args.data_root, split='val')
        eval_coco.run(os.path.join(args.output, 'crf'), args.data_root, split='val')
