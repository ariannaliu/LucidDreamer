#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


import numpy as np
import cv2
def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()

with torch.no_grad():
    kernelsize=3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize//2))
    kernel = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]).reshape(1,1,kernelsize,kernelsize)
    conv.weight.data = kernel #torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()


def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None,None])
    cnt_map = conv((cnt_map * mask)[None,None])
    nearMean_map = (nearMean_map / (cnt_map+1e-8)).squeeze()
        
    return nearMean_map


from torchvision import models
from torch import Tensor
from typing import List

class PerceptualLoss(torch.nn.Module):

    def __init__(self, model='vgg16', device: torch.device=None, weigths=None) -> None:
        super(PerceptualLoss, self).__init__()

        assert model.startswith('vgg'), f"Use vgg model."
        self._vgg = getattr(models, model)(pretrained=True)
        if device is not None:
            self._vgg = self._vgg.to(device)
        self.layers = [4, 9, 16, 23]
        self.weights = weigths if weigths is not None else [1] * len(self.layers)
        self.loss_network = self._vgg.features[:self.layers[-1] + 1].eval()
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = torch.nn.L1Loss()

    def _extract_features(self, x: Tensor) -> List[Tensor]:
        x_vgg = []
        for layer in self.loss_network:
            x = layer(x)
            x_vgg.append(x)
        return x_vgg

    def _gram_mat(self, x: Tensor):
        n, c, h, w = x.shape
        features = x.reshape(n, c, h * w)
        features = features / torch.norm(features, dim=1, keepdim=True) / (h * w) ** 0.5
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        return gram

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        if device is not None:
            assert isinstance(device, torch.device)
            self._vgg.to(device)
            self.loss_network.to(device)
        if dtype is not None:
            assert isinstance(dtype, torch.dtype)
            self._vgg.to(dtype)
            self.loss_network.to(dtype)
        return self

    @property
    def dtype(self):
        return list(self._vgg.parameters())[0].dtype

    def forward(self, out_images: Tensor, target_images: Tensor) -> Tensor:
        out_images = out_images.to(self.dtype)
        target_images = target_images.to(self.dtype)

        input_features, target_features = self._extract_features(out_images), \
                                          self._extract_features(target_images)
        percep_loss = 0
        for weight, layer in zip(self.weights, self.layers):
            loss = weight * self.criterion(input_features[layer].float(), target_features[layer].float())
            # loss = weight * self.criterion(self._gram_mat(input_features[layer]).float(), self._gram_mat(target_features[layer]).float())
            if not (torch.isnan(loss) or torch.isinf(loss)):
                percep_loss += loss

        style_loss = 0
        for weight, layer in zip(self.weights, self.layers):
            loss = weight * self.criterion(self._gram_mat(input_features[layer]).float(), self._gram_mat(target_features[layer]).float())
            if not (torch.isnan(loss) or torch.isinf(loss)):
                style_loss += loss

        return percep_loss, style_loss