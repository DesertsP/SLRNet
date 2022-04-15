import torch
from typing import Tuple, List, Union, Dict, cast, Optional
from torch.distributions import Bernoulli
import torch.nn.functional as F


def resize_as(x, y):
    """
    rescale x to the same size as y
    """
    return F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=True)


def resize_to(x, size):
    """
    rescale x to the same size as y
    """
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


def _adapted_sampling(
    shape: Union[Tuple, torch.Size],
    dist: torch.distributions.Distribution,
    same_on_batch=False
) -> torch.Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.sample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
    else:
        return dist.sample(shape)


def random_prob_generator(
        batch_size: int, p: float = 0.5, same_on_batch: bool = False,
        device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    r"""Generate random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability to generate an 1-d binary mask. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        torch.Tensor: parameters to be passed for transformation.
            - probs (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    if not isinstance(p, (int, float)) or p > 1 or p < 0:
        raise TypeError(f"The probability should be a float number within [0, 1]. Got {type(p)}.")

    _bernoulli = Bernoulli(torch.tensor(float(p), device=device, dtype=dtype))
    probs_mask: torch.Tensor = _adapted_sampling((batch_size,), _bernoulli, same_on_batch).bool()

    return probs_mask


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The horizontally flipped image tensor

    """
    w = input.shape[-1]
    return input[..., torch.arange(w - 1, -1, -1, device=input.device)]


def random_hflip(input: torch.Tensor, p=0.5, return_p=False):
    batch_size, _, h, w = input.size()
    output = input.clone()
    if torch.is_tensor(p):
        to_apply = p
    else:
        to_apply = random_prob_generator(batch_size, p=p)
    output[to_apply] = hflip(input[to_apply])
    if return_p:
        return output, to_apply
    return output


if __name__ == '__main__':
    # print(random_prob_generator(10))
    x = (torch.randn(2,3,5,5))
    print(x[0])
    x_fp, p = random_hflip(x, return_p=True)
    print(x_fp[0])
    x_rec = random_hflip(x_fp, p=p)
    print(x_rec[0])
