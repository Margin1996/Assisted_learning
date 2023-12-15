from PIL import Image
from PIL import ImageEnhance
import glob
import random
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from config import config
import torch
import numpy as np
def random_roate(img1, img2, mask):
    angle = tf.RandomRotation.get_params([-180, 180])
    img1 = img1.rotate(angle)
    img2 = img2.rotate(angle)
    mask = mask.rotate(angle)
    return img1,img2, mask

def enhance_feature(image):
    if random.random() > 0.5:
        enh_image = ImageEnhance.Brightness(image)
        brightness = 1.5
        image = enh_image.enhance(brightness)
    if random.random() > 0.5: 
        enh_col = ImageEnhance.Color(image)
        color = 1.5
        image = enh_col.enhance(color)
    if random.random() > 0.5:
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image = enh_con.enhance(contrast)
    return image

class MyDataset(Dataset):
    def __init__(self, root, is_training=False):
        self.is_training = is_training
        self.root = root
        self.files_A = sorted(glob.glob(os.path.join(root, 'optical') + '/*.tif')) #optical
        self.files_B = sorted(glob.glob(os.path.join(root, 'sar') + '/*.tif')) #SAR
        self.files_D = sorted(glob.glob(os.path.join(root, 'label') + '/*.tif')) #label
        self.trans = tf.Compose([
                tf.ToTensor(),
                tf.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        self.tans_gray = tf.Compose([
                tf.ToTensor(),
                tf.Normalize([0.5], [0.5])
            ])
        # self.img_size = [128,192,256,320,384,448]
        # self.img_size = [128,192,256]
        self.size = config.image_size
        self.num_classes = config.classnum
    def __getitem__(self, index):
        img1 = Image.open(self.files_A[index % len(self.files_A)])
        img2 = Image.open(self.files_B[index % len(self.files_B)])
        # mask = Image.open(self.files_D[index % len(self.files_D)])
        mask = Image.fromarray(cv2.imread(self.files_D[index % len(self.files_D)],0))
        
        if self.is_training:
            img1 = tf.Resize((self.size,self.size))(img1)
            img2 = tf.Resize((self.size,self.size))(img2)
            mask = tf.Resize((self.size,self.size))(mask)

            img1,img2,mask = random_roate(img1,img2, mask)
            img1 = enhance_feature(img1)
            # img2 = enhance_feature(img2)
        else:
            img1 = tf.Resize((self.size,self.size))(img1)
            img2 = tf.Resize((self.size,self.size))(img2)
            mask = tf.Resize((self.size,self.size))(mask)

        img_RGB = np.array(img1)[...,:-1]
        Nir = np.array(img1)[...,-1]
        img_RGB = self.trans(img_RGB)
        Nir = self.tans_gray(Nir)
        image1 = torch.cat([img_RGB,Nir],dim=0)

        image2 = self.tans_gray(img2)
        # mask_img = tf.ToTensor()(mask)
        # mask = np.array(mask)  #dsm rgbn
        mask = np.array(mask)//10 #sar rgbn
        seg_labels  = np.eye(self.num_classes)[mask.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.size), int(self.size), self.num_classes))
        mask = torch.from_numpy(np.array(mask)).long()
        seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
        # return image1, image2, mask, seg_labels
        return img_RGB, image2, mask, seg_labels #only RGB
    def __len__(self):
        return len(self.files_A)
