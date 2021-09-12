import torch
import argparse
import sys
from util_latent import *
sys.path.append('/data/suparna/workspace/vogue/stylegan2-ada-pytorch')
import dnnlib
from identity_garment_loss_sup import *
import pickle
import copy
from training.networks import *
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import copy
import torch.nn.functional as F

class Interpolator(torch.nn.Module):
    def __init__(self, shape, device):
        super().__init__()
        #self.Q = nn.init.uniform_(torch.empty(shape, dtype=torch.float32, device=device, requires_grad=True))
        self.Q = torch.nn.Parameter(torch.rand(shape, dtype=torch.float32, device=device))

    def forward(self, w_p, w_g):
        w_t = self.Q * w_p + (1-self.Q) * w_g
        return w_t

class Optimizer:

    def __init__(self, G, person, garment ):
        self.device = torch.device('cuda')
        self.person = person
        self.garment = garment
        self.G = G
        self.Q = torch.rand(self.person['latent'].shape, dtype=torch.float32, device=self.device, requires_grad=True)
        self.interpolator = Interpolator(self.person['latent'].shape, device=self.device)
        self.interpolator.cuda()
        pose = self.person['pose']
        self.pose = torch.FloatTensor(pose).unsqueeze(0).to(self.device)
        #self.m = F.Sigmoid()

    def printDim(self,temp):
        for t in temp:
            print(t.shape)


    def optimise(self, num_steps=2000):
        alphas = [0.01, 1, 1.0]
        optimizer = torch.optim.Adam(self.interpolator.parameters(), lr=0.005)
        w_avg = self.G.mapping.w_avg
        latent_p = w_avg + (self.person['latent'].clone() -w_avg)*.5
        latent_g = self.garment['latent'].clone()
        
        Q = self.Q
        pose = self.G.encoder(self.pose)
        for step in range(num_steps):
            latent_t = self.interpolator(latent_p, latent_g)
            image = self.G.synthesis(latent_t, pose, noise_mode='const')
            img_t = image[0, 0:3, :, :].permute(1,2,0).clone()
            mask_t = image[0, 3:, :, :].permute(1,2,0).clone()
            ################ loss calculation ################
        
            #### calculate garment loss
            L_garment = garment_loss_tensor(self.garment['img'], self.garment['mask'], img_t, mask_t)
            print('garment loss', L_garment)

            ##### calculate identity loss
            L_identity = identity_loss_tensor(self.person['img'], self.person['mask'], img_t, mask_t)
            print('identity loss', L_identity)

            ##### total loss
            # loss =  alphas[2] * L_identity + alphas[1] * L_garment #+ self.alphas[0] * self.L_loc 
            loss =  L_identity +  L_garment

            print('loss',loss)

            # print(Q.shape)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            print('at step--',step)
            #print(Q[0])
            ########### clip q to [0,1]
            # for qq in range(len(Q)):
            #     Q[qq] = F.sigmoid(Q[qq])


            ################ save image at steps ###########            
            if step % 10 == 0: 
                img_t = (img_t * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
                mask_t = (mask_t * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
                img_t = add_face_to_try_on(img_t, self.person['org_img'], self.person['mask'],mask_t)
                Image.fromarray(img_t, 'RGB').save('out/outit2.png')
                Image.fromarray(mask_t, 'RGB').save('out/outst2.png')
                print('image saved')

        return Q
            
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default='checkpoints/network-snapshot-004939.pkl')
    parser.add_argument("--I_p_org", default='Interpolation/person/targetImg.png')
    parser.add_argument("--I_p", default='Interpolation/person/projImg.png')
    parser.add_argument("--I_g", default='Interpolation/garment2/projImg.png')
    parser.add_argument("--S_p", default='Interpolation/person/projMask.png')
    parser.add_argument("--S_g", default='Interpolation/garment2/projMask.png')
    parser.add_argument("--pose_p", default='Interpolation/person/keypoints.json')
    parser.add_argument("--pose_g", default='Interpolation/garment2/keypoints.json')
    parser.add_argument("--latent_p", default='Interpolation/person/projected_w.npz')
    parser.add_argument("--latent_g", default='Interpolation/garment2/projected_w.npz')
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    img_p, mask_p, pose_p, latent_p = loadData(args.I_p, args.S_p, args.pose_p, args.latent_p)
    img_g, mask_g, pose_g, latent_g = loadData(args.I_g, args.S_g, args.pose_g, args.latent_g)
    org_img = np.array(Image.open(args.I_p_org))
    person = {
        'org_img': org_img,
        'img': img_p,
        'mask': mask_p,
        'pose': pose_p,
        'latent': latent_p
    }
    garment = {
        'img': img_g,
        'mask': mask_g,
        'pose': pose_g,
        'latent': latent_g
    }

    print('Loading networks from "%s"...' % args.network)
    device = torch.device('cuda')
    with open(args.network, 'rb') as f:
        G = pickle.load(f)['G_ema'].requires_grad_(False).to(device)  # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    print('Optimising q')
    opt = Optimizer(G, person, garment)
    opt.optimise(num_steps=2000)

def add_face_to_try_on(try_on, person_img, person_mask, try_on_mask):
    mask1 = np.alltrue(person_mask  == (0, 0, 255), axis=2)
    mask2 = np.alltrue(person_mask == (0, 119, 221), axis=2)
    mask3 = np.where(try_on_mask[:,:,0] < 10, True, False) #R<10 
    mask4 = np.where(try_on_mask[:,:,2] > 200, True, False) #B>200
    mask3 = mask3 * mask4
    mask_t = mask1 + mask2 + mask3
    try_on[mask_t] = person_img[mask_t]
    return try_on


if __name__ == '__main__':
    main()
    
