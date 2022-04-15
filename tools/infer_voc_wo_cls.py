import init_path
import os
import numpy as np
from PIL import Image, ImagePalette
import torch
from torch import multiprocessing
import torch.nn.functional as F
from utils.dcrf import crf_inference
from utils import config_parser
import argparse
from torch.utils.data import DataLoader, Subset
from models import get_model
from datasets import pascal_voc_ms_v2
from utils.utils import denorm
import imageio
from utils.meters import Progbar
import eval_voc


def load_model_weights(model, weight_path):
    assert os.path.isfile(weight_path), 'resume checkpoint not found.'
    checkpoint = torch.load(weight_path, map_location={'cuda:0': 'cpu'})
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    # model.load_state_dict(
    # {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
    # model.load_state_dict()
    return model


class ResultWriter:
    def __init__(self, palette, output_path, apply_crf=True, verbose=False):
        self.palette = palette
        self.root = output_path
        self.verbose = verbose
        self.apply_crf = apply_crf

    @staticmethod
    def colorize_segmentation(segmentation, palette):
        seg_color = Image.fromarray(segmentation.astype(np.uint8))
        seg_color = seg_color.convert('P')
        seg_color.putpalette(palette)
        return seg_color

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
        pred = np.argmax(masks, 0)

        filepath = os.path.join(self.root, 'pred', img_name + '.png')
        img_to_save = self.colorize_segmentation(pred, self.palette)
        img_to_save.save(filepath)

        # CRF
        # converting original image to [0, 255]
        if self.apply_crf:
            img_orig255 = np.round(255. * img_orig).astype(np.uint8)
            img_orig255 = np.transpose(img_orig255, [1, 2, 0])
            img_orig255 = np.ascontiguousarray(img_orig255)
            # pred_crf = crf_inference(img_orig255, masks, t=10, scale_factor=1, labels=21)
            pred_crf = crf_inference(img_orig255, masks, t=10, scale_factor=1, labels=21)
            pred_crf = np.argmax(pred_crf, 0)

            filepath = os.path.join(self.root, "crf", img_name + '.png')
            img_to_save = self.colorize_segmentation(pred_crf, self.palette)
            img_to_save.save(filepath)

        if self.verbose:
            assert self.apply_crf
            mask_gt = gt_mask.numpy()
            masks_all = np.concatenate([pred, pred_crf, mask_gt], 1).astype(np.uint8)
            images = np.concatenate([img_orig] * 3, 2)
            images = np.transpose(images, [1, 2, 0])

            overlay = self._mask_overlay(masks_all, images)
            filepath = os.path.join(self.root, "../vis", img_name + '.png')
            overlay255 = np.round(overlay * 255.).astype(np.uint8)
            imageio.imsave(filepath, overlay255)


def _worker(process_id, model, datasets, scales, output_path, num_workers=4, verbose=1, bg_pow=3, fp_cut=0.1,
            use_cls_label=True, co_clustering=False, apply_crf=True, topk_classes=6):
    databin = datasets[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=num_workers, pin_memory=False)
    writer = ResultWriter(palette=databin.dataset.get_palette(), output_path=output_path, apply_crf=apply_crf,
                          verbose=verbose)
    pool = multiprocessing.Pool(processes=num_workers)
    progbar = Progbar(len(data_loader), prefix="infer", verbose=1)
    with torch.no_grad(), torch.cuda.device(process_id):
        model.cuda()
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            mask = pack['mask'][0]
            size = pack['size']
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
            masks_pred = torch.sum(torch.stack(out_cam_list, 0), 0) / (2 * len(out_cam_list))
            cls_pred = torch.sum(torch.stack(cls_list, 0), 0) / len(cls_list)

            image_with_origin_size = pack['img'][np.where(np.array(scales) == 1)[0][0]][0][0]
            image = denorm(image_with_origin_size).cpu().numpy()
            masks_pred = masks_pred.cpu()
            masks_pred = masks_pred.numpy()

            if num_workers > 0:
                pool.apply_async(writer.save, args=(img_name, image, masks_pred, mask))
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
    parser.add_argument('--bg_pow', type=float, default=3, help='bg_pow=3 for wss, bg_pow=1 for sss')
    parser.add_argument('--use_cls_label', type=int, default=1)
    parser.add_argument('--fp_cut', type=float, default=0.3)
    parser.add_argument('--cluster_iters', type=int, default=1)
    parser.add_argument('--co_cluster', type=int, default=0)
    parser.add_argument('--apply_crf', type=int, default=1)
    parser.add_argument('--topk_classes', type=int, default=6)
    parser.add_argument('--data_list', type=str, default='voc12/val.txt')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    NUM_WORKERS = 4
    print(vars(args))
    config = config_parser.parse_config(args.config, args.opts)
    print("multi-scale:", config.test.scales)
    if os.path.basename(args.data_list).startswith('test'):
        # no ground-truth provided
        test_mode = True
        dataset = pascal_voc_ms_v2.VOC12ClassificationDatasetTest(args.data_list,
                                                                  voc12_root=args.data_root,
                                                                  scales=config.test.scales)
    elif os.path.basename(args.data_list).startswith('train_aug'):
        test_mode = True
        dataset = pascal_voc_ms_v2.VOC12ClassificationDatasetMSF(args.data_list,
                                                                 voc12_root=args.data_root,
                                                                 scales=config.test.scales)
    else:
        test_mode = False
        dataset = pascal_voc_ms_v2.VOC12ClassificationDatasetMSF(args.data_list,
                                                                 voc12_root=args.data_root,
                                                                 scales=config.test.scales)
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

    multiprocessing.spawn(_worker, nprocs=n_gpus,
                          args=(
                              model, datasets, config.test.scales, args.output, NUM_WORKERS, args.verbose, args.bg_pow,
                              args.fp_cut, args.use_cls_label, args.co_cluster, args.apply_crf, args.topk_classes),
                          join=True)

    torch.cuda.empty_cache()
    if not test_mode:
        eval_voc.run(os.path.join(args.output, 'pred'), args.data_root,
                     split=os.path.basename(args.data_list).split('.')[0])
        if args.apply_crf:
            eval_voc.run(os.path.join(args.output, 'crf'), args.data_root,
                         split=os.path.basename(args.data_list).split('.')[0])
