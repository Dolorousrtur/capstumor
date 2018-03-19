import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import itertools

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import img_as_float
from skimage.exposure import equalize_hist
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import block_reduce
from sklearn.utils import shuffle
import pandas as pd
from os import listdir

import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from tqdm import tqdm_notebook

import h5py

def read_h5py_folder(path, indicies=(0,10)):
    columns = ['label', 'PID', 'image', 'tumorBorder', 'tumorMask']
    tmp_df = pd.DataFrame(data = np.zeros((indicies[1]-indicies[0], len(columns))), columns=columns, dtype=np.object)
    files = listdir(path)
    
    for i, each_file in enumerate(files[indicies[0]:indicies[1]]):
        if each_file[-3:]=='mat':
            data = h5py.File(path + '/' + each_file)['cjdata']
            tmp_df['label'][i] = data['label'][0][0]
            tmp_df['PID'][i] = data['PID'][0][0]
            tmp_df['image'][i] = img_as_float(data['image'][:])
            tmp_df['tumorBorder'][i] = data['tumorBorder'][0]
            tmp_df['tumorMask'][i] = img_as_float(data['tumorMask'][:])
        else:
            print('not .mat file dropped')
            tmp_df['label'][i], tmp_df['PID'][i], tmp_df['image'][i] = np.NaN, np.NaN, np.NaN
            tmp_df['tumorBorder'][i], tmp_df['tumorMask'][i] = np.NaN, np.NaN
    tmp_df.dropna(axis=0, inplace=True)
    tmp_df.reset_index(inplace=True)
    return tmp_df



def augment(image, mask, flip_p1=0.7, flip_p2=0.7, factor=2):
    
    img_mask = np.stack([image, mask], axis=2)
    
    def elastic_transform(image, alpha, sigma, alpha_affine):
    
        random_state = np.random.RandomState(None)
        shape = image.shape
        shape_size = shape[:2]

        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    transformed = elastic_transform(img_mask, img_mask.shape[1] * factor, 0.08*img_mask.shape[1], 0.08*img_mask.shape[1])
    
    if np.random.rand() < flip_p1:
        transformed = np.flip(transformed, axis=0)
    
    if np.random.rand() < flip_p2:
        transformed = np.flip(transformed, axis=1)
    
    out_img = transformed[...,0]
    out_mask = transformed[...,1]
        
    return img_as_float(out_img), img_as_float(out_mask)

def resize_fast(X, size=256):
    X['shape_'] = X['image'].apply(lambda x: x.shape[0])
    mean_size = X['shape_'].mean()
    bad_obs_idx = X[X.shape_ <= mean_size].index.tolist()
    bad_obs = X.iloc[bad_obs_idx,:].drop('shape_', axis=1).reset_index(drop=True).copy()
    good_obs = X.drop(bad_obs_idx, axis=0).drop('shape_', axis=1).reset_index(drop=True).copy()

    while (len(good_obs) > 0) and (not good_obs.image[0].shape[0] <= size):
        good_obs.image= good_obs.image.apply(lambda x: block_reduce(x, (2,2), func=np.max))
        good_obs.tumorMask= good_obs.tumorMask.apply(lambda x: block_reduce(x, (2,2), func=np.max))
    
    while (len(bad_obs) > 0) and (not bad_obs.image[0].shape[0] <= size):
        bad_obs.image= bad_obs.image.apply(lambda x: block_reduce(x, (2,2), func=np.max))
        bad_obs.tumorMask= bad_obs.tumorMask.apply(lambda x: block_reduce(x, (2,2), func=np.max))
        
    out = pd.concat([good_obs,bad_obs],axis=0,ignore_index=True)
    out.reset_index(inplace=True)
    return out

def get_train_test_data(X ,ratio):
    df = shuffle(X.copy()).reset_index(drop=True)
    train_idx = int(np.round(df.shape[0]*ratio))
    train_df = df[train_idx:]
    test_df = df[:train_idx]
    train_df.reset_index(inplace=True, drop = True)
    test_df.reset_index(inplace=True, drop = True)
    try:
        return train_df.drop('index', axis=1), test_df.drop('index', axis=1)
    except:
        return train_df, test_df
    
class tumor_data(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        
        if dataframe.image[0].ndim==2:
            images = dataframe.image.apply(lambda x: x.reshape(1, x.shape[0], x.shape[1]))
            
        self.images = images
        self.labels = dataframe.label.tolist()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        img = torch.from_numpy(self.images[idx])
        target = torch.from_numpy(np.array([self.labels[idx]-1]))
        
        return img, target
    
class tumor_data_np(torch.utils.data.Dataset):
    def __init__(self, images, labels, masks=None):
        
            
        self.images = images
        self.labels = labels
        self.masks = masks
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):        
        img = self.images[idx]        
        if self.masks is not None:
            img = img*self.masks[idx]
        img = torch.from_numpy(img)
        
        target = torch.from_numpy(np.array([self.labels[idx]]))
        
        return img, target
    
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
