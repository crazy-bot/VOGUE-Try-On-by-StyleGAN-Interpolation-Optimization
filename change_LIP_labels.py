import os
from PIL import Image
import numpy as np

input_folder = 'tmp/images_label_LIP/'
output_folder = 'tmp/images_label/'

real_images = 'tmp/images_resized/'
segments_folder = 'tmp/images_segmented/'

images = [i for i in os.listdir(input_folder) if 'vis' not in i]
changes = {'0':0,
           '1':  1,
           '2':  1,
           '5':  4,
           '6':  4,
           '7':  4,
           '18': 5,
           '19': 6,
           '3':  7,
           '8':  7,
           '10': 7,
           '11': 7,
           '9':  8,
           '12': 8,
           '16': 9,
           '17': 10,
           '14': 11,
           '4':  12,
           '13': 12,
           '15': 13
           }


for image in images:
    #convert LIP labels to ACGPN
    c = Image.open(input_folder + image)
    np_im = np.array(c)
    np_im2 = np.copy(np_im)

    for key, value in changes.items():
        indexes = np.where(np_im == int(key))
        np_im2[indexes] = value

    new_im = Image.fromarray(np_im2)
    new_im.save(output_folder + image)

    # get segmentations
    c2 = Image.open(real_images + image.split('.')[0] + '.jpg')
    real_img =np.array(c2)
    np_im3= np.copy(np_im)
    indexes = np.where(np_im == changes['0'])
    real_img[indexes] = 255
    new_im = Image.fromarray(real_img)
    new_im.save(segments_folder + image)




