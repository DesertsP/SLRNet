import init_path
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils.meters import AverageMeter, Progbar
from utils.utils import denorm, reduce_tensor
from utils import config_parser
from utils.log_utils import create_log_dir
import argparse
import torch.backends.cudnn as cudnn
import logging
from models import get_model
from datasets import get_dataset
from utils.visualize import mask_rgb, make_grid
import shutil

cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)


class Trainer(object):
    def __init__(self, model, train_set, val_set, config, local_rank=-1, visualize=True):
        # parallel
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        if self.is_distributed:
            logging.info('Distributed training enabled.')
        self.config = config
        self.lr = config.train.lr
        self.num_epochs = config.train.num_epochs
        self.visualize_cams = visualize

        if config.model.backbone == 'resnet38':
            lr_multi = [1.0, 2.0, 10.0, 20.0]
        else:
            lr_multi = [1.0, 1.0, 10.0, 10.0]
        param_groups = model.parameter_groups()
        param_groups = [
            {'params': param_groups[0], 'lr': lr_multi[0] * self.lr, 'weight_decay': self.config.train.weight_decay},
            {'params': param_groups[1], 'lr': lr_multi[1] * self.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': lr_multi[2] * self.lr, 'weight_decay': self.config.train.weight_decay},
            {'params': param_groups[3], 'lr': lr_multi[3] * self.lr, 'weight_decay': 0},
        ]
        # optimizer
        self.optimizer = torch.optim.SGD(param_groups, lr=self.lr, momentum=0.9,
                                         weight_decay=self.config.train.weight_decay, nesterov=True)

        if self.is_distributed:
            device = torch.device('cuda:{}'.format(local_rank))
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", init_method='env://')
            # model
            model = model.to(device)

            self.model = DistributedDataParallel(model, device_ids=[local_rank],
                                                 output_device=local_rank,
                                                 find_unused_parameters=True)
        else:
            model = model.cuda()
            self.model = nn.DataParallel(model)

        # dataset
        self.sampler = DistributedSampler(train_set) if self.is_distributed else None
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.train.batch_size,
            shuffle=False if self.is_distributed else True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.sampler)
        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config.train.batch_size,
            shuffle=False if self.is_distributed else True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(val_set) if self.is_distributed else None)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       (
                                                                                   self.num_epochs - self.config.train.pretrain_epochs) * len(
                                                                           self.train_loader),
                                                                       eta_min=0.0)
        self.tensorboard = SummaryWriter(config.misc.tensorboard_log_dir) if local_rank <= 0 else None
        self.visualize_samples = None
        if os.path.isfile(config.misc.visualize_samples):
            logging.info(f'loading visualize samples from {config.misc.visualize_samples}')
            self.visualize_samples = torch.load(config.misc.visualize_samples)

    def resume(self):
        assert os.path.isfile(self.config.misc.resume_checkpoint_path), 'resume checkpoint not found.'
        checkpoint = torch.load(self.config.misc.resume_checkpoint_path, map_location={'cuda:0': 'cpu'})
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=True)
        logging.info("=> load checkpoint from {}".format(self.config.misc.resume_checkpoint_path))
        start_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return start_epoch

    def train(self):
        start_epoch = 0
        min_loss = 10e9
        if self.config.misc.resume:
            start_epoch = self.resume()
            start_epoch += 1
            if self.is_distributed:
                torch.distributed.barrier()
        for i in range(start_epoch, self.num_epochs):
            if self.is_distributed:
                self.sampler.set_epoch(i)
            loss = self.train_epoch(i)
            val_loss = self.validation(i)
            if self.local_rank <= 0:
                if self.visualize_cams:
                    self.visualize(i)
                if i > 15:
                    torch.save({'state_dict': self.model.module.state_dict(), 'epoch': i,
                                'optimizer': self.optimizer.state_dict()},
                               os.path.join(config.misc.checkpoint_dir, f'checkpoint{i}.pth.tar'))
            if val_loss <= min_loss:
                min_loss = val_loss
                if self.local_rank <= 0:
                    torch.save({'state_dict': self.model.module.state_dict(), 'epoch': i,
                                'optimizer': self.optimizer.state_dict()},
                               os.path.join(config.misc.checkpoint_dir, 'best.pth.tar'))

    def train_epoch(self, epoch):
        progbar = Progbar(len(self.train_loader), prefix="train[{}/{}]".format(epoch + 1, self.num_epochs),
                          verbose=self.config.misc.verbose) if self.local_rank <= 0 else None

        for step, data in enumerate(self.train_loader):
            self.model.train()
            global_step = len(self.train_loader) * epoch + step
            img, label, _ = data
            if self.local_rank <= 0 and self.visualize_samples is None:
                self.visualize_samples = (img.clone(), label.clone())
            img_raw = denorm(img.clone())
            label = label.cuda()
            cls_loss, seg_loss, reg_loss1, reg_loss2, masks, _ = self.model(img.cuda(), img_raw.cuda(), label)

            cls_loss = cls_loss.mean()
            seg_loss = seg_loss.mean()
            reg_loss1 = reg_loss1.mean()
            reg_loss2 = reg_loss2.mean()
            loss = config.train.loss_coeff[0] * cls_loss + \
                   config.train.loss_coeff[1] * reg_loss1 + \
                   config.train.loss_coeff[2] * reg_loss2
            if epoch > self.config.train.pretrain_epochs:
                loss += config.train.loss_coeff[3] * seg_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch > self.config.train.pretrain_epochs:
                self.lr_scheduler.step()

            reduced_loss = reduce_tensor(loss) if self.is_distributed else loss

            if self.local_rank <= 0:
                progbar.update(step + 1,
                               values=[("loss", reduced_loss.item()),
                                       ("cls", cls_loss.item()),
                                       ("seg", seg_loss.item()),
                                       ("reg1", reg_loss1.item()),
                                       ('reg2', reg_loss2.item())])

            if self.local_rank <= 0 and (global_step + 1) % self.config.misc.log_freq == 0:
                self.tensorboard.add_scalars("lr", {"lr": self.optimizer.param_groups[0]['lr']}, global_step)
                self.tensorboard.add_scalars("loss_train", {"loss": progbar['loss'], "seg": progbar['seg'],
                                                            "cls": progbar['cls'], "reg1": progbar['reg1'],
                                                            "reg2": progbar['reg2']}, global_step)
        return progbar['loss'] if self.local_rank <= 0 else 0

    def visualize(self, step):
        """
        :param step:
        :param image: (B, 3, H, W), normed
        :param recon: (B, 3, H, W), normed
        :param masks:  (B, K, H, W), normed
        :param objects: (B, K, 3, H, W)
        :return:
        """
        assert self.visualize_samples is not None
        image, label = self.visualize_samples
        self.model.eval()
        image_raw = denorm(image.clone())
        with torch.no_grad():
            masks = self.model(image.cuda(), image_raw.cuda(), label.cuda())[-2]
        vis_list = [image_raw, mask_rgb(masks['cam'], image_raw),
                    mask_rgb(masks['dec'], image_raw)]
        pseudo_mask = mask_rgb(masks["pseudo"], image_raw)
        ambiguous = 1 - pseudo_mask.sum(1, keepdim=True).cpu()
        pseudo_mask = ambiguous * image_raw + (1 - ambiguous) * pseudo_mask
        vis_list.append(pseudo_mask)

        vis_elements = torch.cat(vis_list, dim=-1)
        # print(label.shape)
        summary_grid = make_grid(vis_elements, label)
        self.tensorboard.add_image('vis', summary_grid, step)

    def validation(self, epoch):
        progbar = Progbar(len(self.val_loader), prefix="valid[{}/{}]".format(epoch + 1, self.num_epochs),
                          verbose=self.config.misc.verbose) if self.local_rank <= 0 else None
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_loader):
                img, label, _ = data
                img_raw = denorm(img.clone())
                label = label.cuda()
                cls_loss, seg_loss, reg_loss1, reg_loss2, masks, slots_update = self.model(img.cuda(), img_raw.cuda(),
                                                                                           label)
                cls_loss = cls_loss.mean()
                seg_loss = seg_loss.mean()
                reg_loss1 = reg_loss1.mean()
                reg_loss2 = reg_loss2.mean()
                loss = cls_loss + reg_loss1 + reg_loss2
                if epoch > self.config.train.pretrain_epochs:
                    loss += seg_loss
                reduced_loss = reduce_tensor(loss) if self.is_distributed else loss
                if self.local_rank <= 0:
                    progbar.update(step + 1,
                                   values=[("loss", reduced_loss.item()), ("cls", cls_loss.item()),
                                           ("seg", seg_loss.item()),
                                           ("reg1", reg_loss1.item()), ('reg2', reg_loss2.item())])

        if self.local_rank <= 0:
            self.tensorboard.add_scalars("loss_val",
                                         {"loss": progbar['loss'], "seg": progbar['cls'], "cls": progbar['cls'],
                                          "reg1": progbar['reg1'], "reg2": progbar['reg2']},
                                         epoch + 1)
        return progbar['cls'] if self.local_rank <= 0 else 0


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=128)
    parser.add_argument('--run', type=str, default='', help="running ID")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = config_parser.parse_config(args.config, args.opts)
    if args.local_rank <= 0:
        log_dir, tensorboard_log_dir, checkpoint_dir = create_log_dir(config.misc.log_dir,
                                                                      os.path.basename(args.config).split('.')[0],
                                                                      run_name=args.run)
        config.misc.log_dir = log_dir
        config.misc.tensorboard_log_dir = tensorboard_log_dir
        config.misc.checkpoint_dir = checkpoint_dir
        print(config)
        # backup models and other scripts
        if os.path.exists(os.path.join(log_dir, 'models')):
            shutil.rmtree(os.path.join(log_dir, 'models'))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
        shutil.copy(args.config, log_dir)
        shutil.copy(__file__, log_dir)

    if args.seed > 0:
        print('Seeding with', args.seed)
        torch.manual_seed(args.seed)

    model = get_model(**config.model)
    # loss_function = nn.MultiLabelSoftMarginLoss()
    # model = FullModel(model, loss_function)
    train_set = get_dataset(**config.dataset, split='train')
    val_set = get_dataset(**config.dataset, split='val')
    engine = Trainer(model, train_set=train_set,
                     val_set=val_set, config=config, local_rank=args.local_rank)
    engine.train()
