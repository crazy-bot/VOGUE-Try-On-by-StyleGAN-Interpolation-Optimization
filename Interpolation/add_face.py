import cv2
import numpy as np

def add_face_to_try_on(try_on, person_mask, face_img):
    mask1 = np.alltrue(person_mask  == (0, 0, 255), axis=2)
    mask2 = np.alltrue(person_mask == (0, 119, 221), axis=2)
    mask_t = mask1 + mask2
    mask = np.where(mask_t > 0, 1, 0)
    binary_mask = np.where((mask == True), 1, 0).astype('uint8')
    face_img_ = cv2.bitwise_and(face_img, face_img, mask=binary_mask)
    face_try_on = face_img_ + try_on
    return face_try_on


if __name__ == '__main__':
    try_on = cv2.imread('out/outit2.png')
    person_mask = cv2.imread('person/10.png')
    face_img = cv2.imread('person/target.png')
    face_try_on = add_face_to_try_on(try_on, person_mask, face_img)
    cv2.imwrite('out/face.png', face_try_on)

