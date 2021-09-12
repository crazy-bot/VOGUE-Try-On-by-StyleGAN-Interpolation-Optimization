import cv2
import numpy as np
from PIL import Image
img = cv2.imread('1_0.png')
img = Image.open('1_0.png').convert('RGB')
img = np.array(img)
h,w, _ = img.shape
w = int(w/2)
print(h,w)

img = img[:,w-350:w+350,:]
mask = np.alltrue(img == (0,0,0), axis=2)
img[mask] = (255,255,255)
img = Image.fromarray(img)
img = img.resize((256,256))
img.save('1_0_resized.png')
exit()
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,_ = img.shape
w = int(w/2)
print(h,w)

img = img[:,w-400:w+400,:]
img = cv2.resize(img,(512,512))
cv2.imshow('p',img)
cv2.waitKey(0)
cv2.imwrite('1_0_resized.png', img)
person_img = cv2.imread('Interpolation/167/targetImg.png')
person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
mask1 = np.alltrue(person_mask  == (0, 0, 255), axis=2)
mask2 = np.alltrue(person_mask == (0, 119, 221), axis=2)
mask_t = mask1 + mask2
img = person_img[mask_t]
cv2.imshow('p',person_mask)
cv2.waitKey(0)