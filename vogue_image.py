import os
from PIL import Image
import numpy as np

mask_folder = '/data/suparna/Data/soccershirt/BM_mask'
maskvis_folder = '/data/suparna/Data/soccershirt/BM_maskvis'
image_folder = '/data/suparna/Data/soccershirt/BM_ready'
vogue_image = '/data/suparna/Data/soccershirt/BM_small'
vogue_mask = '/data/suparna/Data/soccershirt/BM_voguemask'
LIPTOVOGUE = {
    'background': [0],
    'tops': [5,6,7,11], 
    'bottoms': [8,9,10,12], 
    'face': [13], 
    'hair': [1,2], 
    'arms': [3, 14, 15], 
    'skin': [85,51,0], 
    'legs': [16,17,18,19], 
    'other':[]
}
colorspace = [(255, 255, 255), (255, 85, 0), (85, 85, 0), (0, 0, 255), (0, 119, 221), (51, 170, 221), (85,51,0), (170, 255, 85), (52, 86, 128)]
print(colorspace[0])
for f in os.listdir(image_folder):
    # maskpath = os.path.join(mask_folder, str(i)+'.png')
    # im_parse = Image.open(maskpath).convert('L')
    # parse_array = np.array(im_parse)
    im_parse_rgb = np.array(Image.open(os.path.join(vogue_mask, f)))    
    img = np.array(Image.open(os.path.join(image_folder, f)))
    # new_mask = np.zeros((512,512,3))
    
    for j, key in enumerate(LIPTOVOGUE.keys()):
        if key in ['tops']: continue
        img[(im_parse_rgb == colorspace[j]).all(-1)] = (255, 255, 255)

        # LIPlabels = LIPTOVOGUE[key]
        # for label in LIPlabels:
        #     #indexes = np.where(parse_array == int(label))
        #     #img[parse_array == int(label)] = colorspace[j]
        #     img[(im_parse_rgb == colorspace[j]).all(-1)] = (255, 255, 255)
            
    im = Image.fromarray(img.astype(np.uint8))
    im.save(os.path.join(vogue_image, f))
    print(os.path.join(vogue_image, f))
    #exit()



