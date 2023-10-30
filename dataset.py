from PIL import Image
from PIL import ImageEnhance
import glob
import random
import os
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
from torch.utils.data import Dataset
import torchvision.transforms as tf
import scipy.signal
import torchvision.transforms.functional as F
from config import config
import torch
import numpy as np
def random_roate(img1, img2, mask):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = tf.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
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
        self.files_A = sorted(glob.glob(os.path.join(root, 'optical') + '/*.tif')) #RGBN
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
        return img_RGB, image2, mask, seg_labels #仅适用RGB


    def __len__(self):
        return len(self.files_A)

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'green', linewidth = 2, linestyle = '--', label='train loss')
        plt.plot(iters, self.val_loss, 'blue', linewidth = 2, linestyle = '--', label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 25
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'red',  linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 'darkorchid', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))
        plt.cla()
        plt.close("all")