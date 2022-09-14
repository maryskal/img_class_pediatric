from tensorflow.keras.utils import Sequence
import numpy as np
import math
import cv2
import os
import funciones_imagenes.prepare_img_fun as fu
import funciones_imagenes.mask_funct as msk
import albumentations as A


def albumentation(input_image):
    transform = A.Compose([
        A.Rotate(limit=90, border_mode = None, interpolation=2, p=1),
        A.OneOf([
            A.RandomCrop(p= 1, width=230, height=230),
            A.GridDistortion (num_steps=5, distort_limit=0.3, interpolation=2, border_mode=None, p=1),
            A.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
            A.ElasticTransform(alpha=0.5, sigma=50, alpha_affine=50, interpolation=1, border_mode=None, always_apply=False, p=1)
        ], p=0.8),
    ])
    transformed = transform(image=input_image.astype(np.float32))
    input_image = fu.get_prepared_img(transformed['image'])
    return input_image


class DataGenerator(Sequence):
    
    def __init__(self, df, batch_size, pix, mask):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size
        self.pix = pix
        self.mask = mask

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.df['real']) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].reset_index(drop = True)
        batch_x = np.zeros((len(batch_df), self.pix, self.pix, 1))
        batch_y = np.array(batch_df[['normal', 'viral', 'bacteria']])
        for i in range(len(batch_df)):
            try:
                img = cv2.imread(os.path.join(batch_df['path'][i], batch_df.img_name[i]))
                batch_x[i,...] = fu.get_prepared_img(img, self.pix, mask = self.mask, clahe_bool=True)
            except:
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = msk.normalize(img)
                print('e')
        # batch_x = fu.augment_tensor(batch_x)
        return batch_x, batch_y


class DataGenerator_augment(Sequence):
    
    def __init__(self, df, batch_size, pix, mask):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size
        self.pix = pix
        self.mask = mask

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.df['real']) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].reset_index(drop = True)
        batch_x = np.zeros((len(batch_df), self.pix, self.pix, 1))
        batch_x_augment = np.zeros((len(batch_df), self.pix, self.pix, 1))
        batch_y = np.array(batch_df[['normal', 'viral', 'bacteria']])
        for i in range(len(batch_df)):
            try:
                img = cv2.imread(os.path.join(batch_df['path'][i], batch_df.img_name[i]))
                batch_x[i,...] = fu.get_prepared_img(img, self.pix, mask = self.mask, clahe_bool=True)
                batch_x_augment[i,...] = albumentation(img)
            except:
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = msk.normalize(img)
                batch_x_augment[i,...] = albumentation(img)
                print('e')

        batch_x = np.concatenate((batch_x, batch_x_augment), axis = 0)
        batch_y = np.concatenate((batch_y, batch_y), axis = 0)
        # batch_x = fu.augment_tensor(batch_x)
        return batch_x, batch_y
