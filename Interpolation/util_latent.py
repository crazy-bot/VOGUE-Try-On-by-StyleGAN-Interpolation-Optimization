import cv2
import numpy as np
import torch
from torchvision import utils
import os
from PIL import Image
import json
import glob
import torchvision
import copy
import os
import json
import shutil
import subprocess
import numpy as np
import sys
import os.path as osp
sys.path.append('/data/suparna/workspace/vogue/stylegan2-ada-pytorch')
import projector_vogue
from torch_utils import misc


colorspace = [(0, 0, 0), (255, 85, 0), (85, 85, 0), (0, 0, 255), (0, 119, 221), (51, 170, 221), (85,51,0), (170, 255, 85), (52, 86, 128)]
transforms = []
transforms.append(torchvision.transforms.ToTensor())
# transforms.append(
#             torchvision.transforms.Normalize(
#                 mean=[-(pixel_min / (pixel_max - pixel_min))],
#                 std=[1. / (pixel_max - pixel_min)]
#             )
#         )
transform = torchvision.transforms.Compose(transforms)

def cords_to_map(keypoints, img_size = (512,512), std = 6):
        heatmap = np.zeros((keypoints.shape[0], img_size[0], img_size[1]))
        
        xx, yy = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))

        for i, point in enumerate(keypoints):            
            # result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
            # result[..., i] = np.where(((yy - point[0]) ** 2 + (xx - point[1]) ** 2) < (sigma ** 2), 1, 0)
            xi, yi, zi = keypoints[i]
            heatmap[i,:,:] = np.exp(-(np.power((xx - xi)/std, 2)/2 + np.power((yy - yi)/std, 2)/2 ))

        return heatmap

def loadData(img_path, mask_path, jsonpath, latent_path):

    device = torch.device('cuda')
    img = Image.open(img_path)
    img = np.array(img)
    #img = torch.Tensor(img)

    mask = Image.open(mask_path)
    mask = np.array(mask).astype('uint8')
    # mask = torch.Tensor(mask)
    # Image.fromarray(mask.numpy().astype('uint8')).save('out/ig_mask.png')  
    # exit()

    keypoints = json.load(open(jsonpath,'r'))
    keypoints = keypoints['people'][0]['pose_keypoints_2d']
    keypoints = np.array(keypoints)
    keypoints = keypoints.reshape((-1, 3))
    heatmaps = cords_to_map(keypoints=keypoints)
    heatmaps = torch.FloatTensor(heatmaps)

    z = np.load(latent_path)
    w = torch.tensor(z['w'], device=device).to(torch.float32)
        
    return img, mask, heatmaps, w

################### start: utility functions to calculate M_p and M_g ##################
def get_U(img, dim):
    # img = cv2.imread(mask_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize(dim, resample= Image.BICUBIC)
    img = np.array(img)
    W,H,C = img.shape    
    heatmaps = torch.zeros((len(colorspace), W, H))
    for i in range(len(colorspace)):
        mask = np.alltrue(img == colorspace[i], axis=2)
        mask_segment = np.where((mask == True), 1, 0).astype('uint8')
        heatmaps[i] = torch.tensor(mask_segment)
    return heatmaps.unsqueeze_(0)


def get_M_per_layer(A, U):
    N, C, H, W = A.shape
    norm_const = 1 / (N*H*W)
    A = A@A
    A = A.reshape(C, N * H * W)
    N, K, H, W = U.shape
    U = U.reshape(K, N * H * W)
    M_per_layer = torch.matmul(U.float().cuda(), A.T.float().cuda())
    return norm_const * M_per_layer


def get_M(A, mask):
    M = []
    for i in range(len(A)):
        _,_, H, W = A[i].shape
        U = get_U(mask, (H,W))
        M.append(get_M_per_layer(A[i], U))
    return M

################### End: utility functions to calculate M_p and M_g ##################

def get_projection(target):
    network_pkl = 'images/network-snapshot-000400.pkl'
    target_fname = 'images/10.png'
    keypoints_json = 'images/pose/10_keypoints.json'
    outdir = 'out_10'
    return projector.run_projection(network_pkl, target_fname, keypoints_json, outdir)

def get_m_i(m):
    M = copy.deepcopy(m)
    M = M.cpu().numpy()
    tmp = np.zeros_like(M[0])
    for i in range(M.shape[0]):
        m_i = M - M[i]
        index = np.argmax(np.max(m_i, axis=1))
        if np.sum(m_i[index]) > np.sum(tmp):
            tmp = m_i[index]
    return tmp

def localization_loss(M_i, Q):
    loss_loc = torch.zeros(1,dtype=torch.float32,device=torch.device('cuda'),requires_grad=True)
    for i in range(len(M_i)):
        # print('M_i[i]', M_i[i].shape)
        # print('Q[i]', Q[i].shape)
        loss = torch.matmul(M_i[i], Q[i].T)
        #print(loss[0])
        loss_loc += loss[0]
        #print('loss_loc',loss_loc)
    return loss_loc

def localisation_loss_per_layer(M_p, M_g):
    m_p_i = get_m_i(M_p)
    m_g_i = get_m_i(M_g)
    m_i = m_g_i if (np.sum(m_g_i) > np.sum(m_p_i )) else m_p_i  # per layer 1x512
    return m_i


class StyleInterpolator:

    def __init__(self, mask_p, pose_p, w_p, mask_g, pose_g, w_g, G=None):
        
        self.device = torch.device('cuda')
        
        ############### Initialize Q ##############
        self.latent_dim = 512
        self.Q = torch.rand(w_p.shape, dtype=torch.float32, device=self.device, requires_grad=True)
        self.w_t = torch.empty(w_p.shape, dtype=torch.float32, device=self.device, requires_grad=True)
        self.act_p, self.act_g = [], []
        self.pose_p = G.encoder(pose_p.unsqueeze_(0).cuda())
        self.pose_g = G.encoder(pose_g.unsqueeze_(0).cuda())

        ############# accumulate activation, style and initialize q for each block ########################
        
        
    def get_interpolated_style(self, w_p, w_g, Q):
        w_t = Q * w_p + (1-Q) * w_g
        return w_t







