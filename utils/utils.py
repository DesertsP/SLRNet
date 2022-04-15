import os
import torch
import torch.distributed as dist

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def adjust_learning_rate(optimizer, base_lr, num_warmup_steps, num_decay_steps, cur_step, decay_rate=0.5):
    """
    param_groups = [
            {'params': param_groups[0], 'lr': self.lr, 'weight_decay': self.config.train.weight_decay},
            {'params': param_groups[1], 'lr': 2.0 * self.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10.0 * self.lr, 'weight_decay': self.config.train.weight_decay},
            {'params': param_groups[3], 'lr': 20.0 * self.lr, 'weight_decay': 0},
        ]
    :param optimizer:
    :param base_lr:
    :param num_warmup_steps:
    :param num_decay_steps:
    :param cur_step:
    :param decay_rate:
    :return:
    """
    if cur_step <= num_warmup_steps:
        lr = (cur_step) * base_lr / num_warmup_steps
    elif cur_step <= num_warmup_steps + num_decay_steps:
        lr = base_lr * decay_rate ** ((cur_step - num_warmup_steps) / num_decay_steps)
    else:
        lr = base_lr * decay_rate
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = 10.*lr
    elif len(optimizer.param_groups) == 4:
        optimizer.param_groups[1]['lr'] = 2. * lr
        optimizer.param_groups[2]['lr'] = 10. * lr
        optimizer.param_groups[3]['lr'] = 20. * lr
    return lr


def denorm(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if image.dim() == 3:
        assert image.dim() == 3, "Expected image [CxHxW]"
        assert image.size(0) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
    elif image.dim() == 4:
        # batch mode
        assert image.size(1) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip((0, 1, 2), mean, std):
            image[:, t, :, :].mul_(s).add_(m)
    else:
        raise NotImplementedError

    return image


if __name__ == '__main__':
    pass
