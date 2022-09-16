from ctypes import sizeof
from tensorflow.keras.utils import Sequence
import numpy as np
import math
import cv2
import os
import sys
import funciones_imagenes.image_funct as fu


def charge_imgs(img_list):
    path = '/home/mr1142/Documents/Data/NIH'
    images = []
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

    # De todas las imagenes que quiero cargar miro cuantas hay en cada carpeta y las cargo
    for folder in subfolders:
        folder_images = os.listdir(os.path.join(folder, 'images'))
        this_folder_imgs = set(img_list).intersection(folder_images)
        for im in this_folder_imgs:
            images.append(cv2.imread(folder + '/images/' + im))

    return images


class DataGenerator(Sequence):
    
    def __init__(self, images_list, batch_size, pix, mask):
        self.batch_size = batch_size
        self.pix = pix
        self.images_list = images_list
        self.mask = mask
        self.errors = 0
        self.errors_list = []

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.images_list) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        batch_images_names = self.images_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        images_list = charge_imgs(batch_images_names)
        batch_x = np.zeros((len(images_list), self.pix, self.pix, 1), dtype = 'float32')
        for i in range(len(images_list)):
            try:
                batch_x[i,...] = fu.get_prepared_img(images_list[i], self.pix, self.mask)
            except:
                self.errors += 1
                self.errors_list.append(idx*self.batch_size+i)
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = fu.normalize(img)

        # batch_x = fu.augment_tensor(batch_x)

        return batch_x, batch_x

