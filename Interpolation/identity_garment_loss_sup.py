import cv2
import numpy as np
import torch
from torchvision import utils
import os
from PIL import Image
import json
import glob
import torchvision
#import dnnlib
from torchvision import models
from vgg16 import *
from PIL import Image
import copy
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
outdir='out'
colorspace = [(0, 0, 0), (255, 85, 0), (85, 85, 0), (0, 0, 255), (0, 119, 221), (51, 170, 221), (85,51,0), (170, 255, 85), (52, 86, 128)]
vgg_model = Vgg16().cuda()

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

smoothing = GaussianSmoothing(3, 5, 1).cuda()

def perceptual_distance(img1, img2):
    weight = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]  # more high level info
    weight.reverse()
    #img1 = img1.permute(0, 3, 1, 2)
    #img2 = img2.permute(0, 3, 1, 2)
    # print(img1.shape)
    # print(img2.shape)
    img1_features = vgg_model(img1.float())
    img2_features = vgg_model(img2.float())
    loss = nn.L1Loss()#.cuda()
    weighted_dist = 0
    for i in range(4):
        weighted_dist = weighted_dist + weight[i] * loss(img1_features[i], img2_features[i])
    return weighted_dist


def get_binary_mask(img, type='garment'):
    #img = cv2.imread(mask_path)
    #mask_path =  img.permute(1,2,0)
    #mask_p = torch.clone(mask_path).cpu().detach().numpy()
    #img = np.array(mask_p).astype(np.uint8)
    #img = img*255
    # if img.shape[0] == 3:
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if type == 'garment':
        mask = np.alltrue(img == (255, 85, 0), axis=2)
    else:
        mask1 = np.alltrue(img == (0, 0, 255), axis=2)
        mask2 = np.alltrue(img == (0, 119, 221), axis=2)
        mask_t = mask1 + mask2
        mask = np.where(mask_t > 0, 1, 0)

    binary_mask = np.where((mask == True), 1, 0).astype('uint8')
    return binary_mask

def garment_loss(s_g, s_t, I_g, I_t):
    S_g_binary = get_binary_mask(s_g) *255
    #print(S_g_binary.shape)# 1, 512, 512
    # I_g = I_g.permute(1, 2, 0)
    # I_g = np.array(I_g).astype(np.uint8)
    # I_g = I_g * 255
    Image.fromarray(S_g_binary.astype(np.uint8)).save('out/Ig_mask.png')
    I_g_masked = cv2.bitwise_and(I_g, I_g, mask=S_g_binary)  # 3, 512, 512
    I_g_masked_blur = cv2.GaussianBlur(I_g_masked, (5, 5), 0)
    I_g_masked_blur_resized = cv2.resize(I_g_masked_blur, (256,256), interpolation=cv2.INTER_AREA)
    I_g_masked_blur_resized = I_g_masked_blur_resized / 255

    print(s_t.shape)
    S_t_binary = get_binary_mask(s_t)
    I_t = I_t.permute(1, 2, 0).cpu().detach().numpy()
    I_t = np.array(I_t).astype(np.uint8)
    I_t = I_t * 255
    I_t_masked = cv2.bitwise_and(I_t, I_t, mask=S_t_binary)
    I_t_masked_blur = cv2.GaussianBlur(I_t_masked, (5, 5), 0)
    I_t_masked_blur_resized = cv2.resize(I_t_masked_blur, (256, 256), interpolation=cv2.INTER_AREA)
    I_t_masked_blur_resized = I_t_masked_blur_resized / 255
    #Image.fromarray(I_g_masked_blur_resized.astype(np.uint8)).save('out/Ig_mask.png')


    d = perceptual_distance(torch.tensor(I_g_masked_blur_resized).unsqueeze_(0), torch.tensor(I_t_masked_blur_resized).unsqueeze_(0))
    return d

def garment_loss_tensor(i_g, s_g, i_t, s_t):
    #print(s_g)
    s_g = torch.Tensor(s_g)
    i_g = torch.Tensor(i_g)
    s_g_mask = torch.where(s_g[:,:,0] >= torch.Tensor([200]), 1, 0)
    #s_g_mask = torch.all(s_g == torch.Tensor([255, 85, 0]),axis =2)
    s_g_mask = s_g_mask.unsqueeze(-1).expand(i_g.size())       
    i_g_masked = i_g * s_g_mask
    #print('i_g_masked', i_g_masked.shape)
    #i_g_masked = F.pad(i_g_masked, (2, 2, 2, 2), mode='reflect')
    # i_g_masked = smoothing(i_g_masked.permute(2,0,1).unsqueeze(0).cuda())
    i_g_masked= F.interpolate(i_g_masked.permute(2,0,1).unsqueeze(0), size=256).cuda()

    # testimg = i_g_masked.to(torch.uint8).clamp(0,255)
    # Image.fromarray(testimg[ :, :, :].cpu().numpy()).save(f'{outdir}/i_g_masked.png')
    # exit()

    #img = (i_g_masked.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # testimg = i_g_masked.permute(0, 2, 3, 1).to(torch.uint8).clamp(0,255)
    # Image.fromarray(testimg[0, :, :, :].cpu().numpy(), 'RGB').save(f'{outdir}/i_g_masked.png')
    # exit()
    #i_g_masked = i_g_masked / torch.max(i_g_masked)

    # s_t_mask = torch.all(s_t == torch.Tensor([255, 85, 0]).cuda(),axis=2)
    #s_t = (s_t * 127.5 + 128).clamp(0, 255)
    s_t = s_g.cuda()
    i_t = (i_t * 127.5 + 128).clamp(0, 255)
    s_t_mask = torch.where(s_t[:,:,0] >= torch.Tensor([200]).cuda(), 1, 0)
    s_t_mask = s_t_mask.unsqueeze(-1).expand(i_t.size())
    i_t_masked = i_t * s_t_mask
    # i_t_masked = smoothing(i_t_masked.permute(2,0,1).unsqueeze(0).cuda())
    i_t_masked= F.interpolate(i_t_masked.permute(2,0,1).unsqueeze(0), size=256)

    # testimg = s_t.to(torch.uint8)
    # Image.fromarray(s_t.cpu().numpy()*255, 'RGB').save(f'{outdir}/i_t_masked.png')
    # i_t_masked = (i_t_masked * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # Image.fromarray(i_t_masked.to(torch.uint8).cpu().numpy(), 'RGB').save(f'{outdir}/i_t_masked.png')
    # exit()

    # testimg = i_t_masked.permute(0, 2, 3, 1).to(torch.uint8).clamp(0,255)
    # Image.fromarray(testimg[0, :, :, :].cpu().numpy(), 'RGB').save(f'{outdir}/i_t_masked.png')
    # exit()
    #i_t_masked = i_t_masked / torch.max(i_t_masked)
    
    d = perceptual_distance(i_t_masked, i_g_masked)
    #print(d)
    return d


def identity_loss_tensor(i_p, s_p, i_t, s_t):
    s_p = torch.Tensor(s_p)
    i_p = torch.Tensor(i_p)

    s_p_mask1 = torch.where(s_p[:,:,2] > torch.Tensor([200]), 1, 0) # arms, face, hair
    
    s_p_mask2 = torch.where(s_p[:,:,0] < torch.Tensor([200]), 1, 0) # bottom , skin
    s_p_mask3 = torch.where(s_p[:,:,1] > torch.Tensor([45]), 1, 0) # bottom , skin
    s_p_mask4 = torch.where(s_p[:,:,1] < torch.Tensor([100]), 1, 0) # bottom , skin

    s_p_mask5 = torch.where(s_p[:,:,0] < torch.Tensor([5]), 1, 0) # background
    s_p_mask = s_p_mask2 * s_p_mask3 * s_p_mask4
    s_p_mask = s_p_mask + s_p_mask1 + s_p_mask5
    # s_p_mask = s_p_mask + s_p_mask5
    s_p_mask = s_p_mask.unsqueeze(-1).expand(i_p.size())
    i_p_masked = i_p * s_p_mask
    # i_g_masked = F.pad(i_g_masked, (2, 2, 2, 2), mode='reflect')
    # i_g_masked = smoothing(i_g_masked.permute(2, 0, 1).unsqueeze(0).cuda())
    i_p_masked = F.interpolate(i_p_masked.permute(2, 0, 1).unsqueeze(0), size=256).cuda()
    # i_p_masked = i_p_masked / torch.max(i_g_masked)
    
    #testimg = s_p.to(torch.uint8)   

    #Image.fromarray(img[:, :, :].cpu().numpy()).save(f'{outdir}/i_g_masked_identity.png')
    #print(torch.max(i_g_masked))  
    
    # mask1 = torch.all(s_t == torch.Tensor([0, 0, 255]).cuda(), axis=2) #face
    # mask2 = torch.all(s_t == torch.Tensor([0, 119, 221]).cuda(), axis=2) #hair
    # mask3 = torch.all(s_t == torch.Tensor([51, 170, 221]).cuda(), axis=2) #arms
    # mask_t = mask1 + mask2 + mask3
    #print(mask_t.shape)
    # s_t = (s_t * 127.5 + 128).clamp(0, 255).cpu()
    s_t = s_p
    i_t = (i_t * 127.5 + 128).clamp(0, 255).cpu()
    s_t_mask1 = torch.where(s_t[:,:,2] > torch.Tensor([200]), 1, 0) # arms, face, hair

    s_t_mask2 = torch.where(s_t[:,:,0] < torch.Tensor([200]), 1, 0) # bottom , skin
    s_t_mask3 = torch.where(s_t[:,:,1] > torch.Tensor([45]), 1, 0) # bottom , skin
    s_t_mask4 = torch.where(s_t[:,:,1] < torch.Tensor([100]), 1, 0) # bottom , skin

    s_t_mask5 = torch.where(s_p[:,:,0] < torch.Tensor([5]), 1, 0) # background
    
    s_t_mask = s_t_mask2 * s_t_mask3 * s_t_mask4
    s_t_mask = s_t_mask + s_t_mask1 + s_t_mask5
    #s_t_mask = s_t_mask + s_t_mask5
    s_t_mask = s_t_mask.unsqueeze(-1).expand(i_t.size())
    i_t_masked = i_t * s_t_mask
    #i_t_masked = smoothing(i_t_masked.permute(2, 0, 1).unsqueeze(0)).cuda()
    i_t_masked = F.interpolate(i_t_masked.permute(2, 0, 1).unsqueeze(0), size=256).cuda()
    # i_t_masked = i_t_masked / torch.max(i_t_masked)

    # testimg = i_t_masked.permute(0, 2, 3, 1).to(torch.uint8).clamp(0,255)
    # Image.fromarray(testimg[0, :, :, :].cpu().numpy(), 'RGB').save(f'{outdir}/i_t_masked.png')
    # exit()

    d = perceptual_distance(i_t_masked, i_p_masked)
    
    return d


def identity_loss(s_g_path, s_t_path, I_g, I_t):
    S_g_binary = get_binary_mask(s_g_path, type='identity')
    I_g = I_g.permute(1, 2, 0)
    I_g = np.array(I_g).astype(np.uint8)
    I_g = I_g * 255
    I_g_masked = cv2.bitwise_and(I_g, I_g, mask=S_g_binary)
    I_g_masked_blur = cv2.GaussianBlur(I_g_masked, (5, 5), 0)
    I_g_masked_blur_resized = cv2.resize(I_g_masked_blur, (256,256), interpolation=cv2.INTER_AREA)
    I_g_masked_blur_resized = I_g_masked_blur_resized / 255

    S_t_binary = get_binary_mask(s_t_path, type='identity')
    I_t = I_t.permute(1, 2, 0).cpu().detach().numpy()
    I_t = np.array(I_t).astype(np.uint8)
    I_t = I_t * 255
    I_t_masked = cv2.bitwise_and(I_t, I_t, mask=S_t_binary)
    I_t_masked_blur = cv2.GaussianBlur(I_t_masked, (5, 5), 0)
    I_t_masked_blur_resized = cv2.resize(I_t_masked_blur, (256, 256), interpolation=cv2.INTER_AREA)
    I_t_masked_blur_resized = I_t_masked_blur_resized / 255

    d = perceptual_distance(torch.tensor(I_g_masked_blur_resized).unsqueeze_(0),
                            torch.tensor(I_t_masked_blur_resized).unsqueeze_(0))

    return d

    class GaussianSmoothing(nn.Module):
        """
        Apply gaussian smoothing on a
        1d, 2d or 3d tensor. Filtering is performed seperately for each channel
        in the input using a depthwise convolution.
        Arguments:
            channels (int, sequence): Number of channels of the input tensors. Output will
                have this number of channels as well.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigma (float, sequence): Standard deviation of the gaussian kernel.
            dim (int, optional): The number of dimensions of the data.
                Default value is 2 (spatial).
        """

        def __init__(self, channels, kernel_size, sigma, dim=2):
            super(GaussianSmoothing, self).__init__()
            if isinstance(kernel_size, numbers.Number):
                kernel_size = [kernel_size] * dim
            if isinstance(sigma, numbers.Number):
                sigma = [sigma] * dim

            # The gaussian kernel is the product of the
            # gaussian function of each dimension.
            kernel = 1
            meshgrids = torch.meshgrid(
                [
                    torch.arange(size, dtype=torch.float32)
                    for size in kernel_size
                ]
            )
            for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                          torch.exp(-((mgrid - mean) / std) ** 2 / 2)

            # Make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)

            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.size())
            kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

            self.register_buffer('weight', kernel)
            self.groups = channels

            if dim == 1:
                self.conv = F.conv1d
            elif dim == 2:
                self.conv = F.conv2d
            elif dim == 3:
                self.conv = F.conv3d
            else:
                raise RuntimeError(
                    'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
                )

        def forward(self, input):
            """
            Apply gaussian filter to input.
            Arguments:
                input (torch.Tensor): Input to apply gaussian filter on.
            Returns:
                filtered (torch.Tensor): Filtered output.
            """
            return self.conv(input, weight=self.weight, groups=self.groups)

    def garment_loss_using_tensors_(s_g, s_t, I_g, I_t):
        I_t = I_t.permute(1, 2, 0)
        s_t = (s_t * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        s_t = s_t.permute(1, 2, 0)
        print(s_t[300, 300])
        p = torch.tensor([0, 255, 85])
        mask = torch.all(s_t == p.cuda(), axis=2)
        outdir = 'out'
        #Image.fromarray(mask.cpu().numpy()).save(f'{outdir}/mask.png')
        # print(mask[mask==True])

        smoothing = GaussianSmoothing(3, 5, 1)
        input = torch.rand(1, 3, 100, 100)
        input = F.pad(input, (2, 2, 2, 2), mode='reflect')
        output = smoothing(input)
        print(output)


if __name__ == "__main__":
    g_path = "images/mask/0.png"
    t_path = "images/mask/0.png"
    g = torch.tensor(())
    garment_loss(g_path, t_path, I_g, I_t)




