import numpy as np
import cv2
import os
import h5py
import random
from math import sqrt

#synthetic data - hollow diamond, cross, diamond
#total shapes being used 2000
#out of that, a subset of 20 for validation.
num_shapes_total = 2000
num_shapes_val = 20
image_size = 64
batch_size = image_size*image_size

hdf5_file_all = h5py.File("complex_elements.hdf5", 'w')
hdf5_file_all.create_dataset("pixels", [num_shapes_total,image_size,image_size,1], np.uint8, compression=9)

hdf5_file_val = h5py.File("complex_elements_val.hdf5",'w')
hdf5_file_val.create_dataset("pixels", [num_shapes_val,image_size,image_size,1], np.uint8, compression=9)



#shape 1 - hollow diamond
shape_1_outsider = np.array([[0,1], [-1,0], [0,-1], [1,0]])
shape_1_insider = np.array([[0.5,0], [0,-0.5], [-0.5,0], [0,0.5]])

#shape 2 - cross
shape_2_outsider = np.array([[1,1.0/3], [1.0/3,1.0/3], [1.0/3,1], [-1.0/3,1], [-1.0/3,1.0/3], [-1,1.0/3], [-1,-1.0/3], [-1.0/3,-1.0/3], [-1.0/3,-1], [1.0/3,-1], [1.0/3,-1.0/3], [1,-1.0/3]])

#shape 3 - diamond
shape_3_outsider = np.array([[0,1], [-1,0], [0,-1], [1,0]])


def get_image(x1,y1,s1,x2,y2,s2,x3,y3,s3):
    img = np.zeros([image_size,image_size,1], np.uint8)
    cv2.fillPoly(img, [(shape_1_outsider*s1+np.array([x1,y1])).astype(np.int32)], 1)
    cv2.fillPoly(img, [(shape_1_insider*s1+np.array([x1,y1])).astype(np.int32)], 0)
    cv2.fillPoly(img, [(shape_2_outsider*s2+np.array([x2,y2])).astype(np.int32)], 1)
    cv2.fillPoly(img, [(shape_3_outsider*s3+np.array([x3,y3])).astype(np.int32)], 1)
    return img

for idx in range(num_shapes_total):

    while True:
        s1 = random.randint(12,16)
        x1 = random.randint(s1+1,image_size-s1-2)
        y1 = random.randint(s1+1,image_size-s1-2)
        s2 = random.randint(12,16)
        x2 = random.randint(s2+1,image_size-s2-2)
        y2 = random.randint(s2+1,image_size-s2-2)
        s3 = random.randint(8,12)
        x3 = random.randint(s3+1,image_size-s3-2)
        y3 = random.randint(s3+1,image_size-s3-2)
        if x1>x2+max(s1,s2) and x2>x3+max(s2,s3): break
    img = get_image(x1,y1,s1,x2,y2,s2,x3,y3,s3)
    hdf5_file_all["pixels"][idx] = img
    if idx < num_shapes_val:
        hdf5_file_val["pixels"][idx] = img


hdf5_file_all.close()
hdf5_file_val.close()


