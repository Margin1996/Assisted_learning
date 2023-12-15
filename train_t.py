import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from loss import Dice_loss,CE_Loss
from torch.autograd import Variable
from dataset import MyDataset
import torch.nn as nn
from config import config
import metric
from hrnet.hrnet import HRnet
import time
from torch.cuda.amp import GradScaler as Gradscaler
from torch.cuda.amp import autocast 
from tqdm import tqdm
scaler = Gradscaler()

traindd = MyDataset(config.trainroot,is_training=True)
traindata = DataLoader(traindd,batch_size=config.batch_size, shuffle=True)
valdata = DataLoader(MyDataset(config.valroot,is_training=False), num_workers=0, batch_size=config.batch_size, shuffle=False)
net = HRnet(in_channel=3,num_classes=config.classnum,backbone='hrnetv2_w32',pretrained=True).cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

iters = len(traindata)
train_size = len(traindata)
val_size = len(valdata)
print('train data size: %04d'%train_size)
print('val data size: %04d'%val_size)
global_Fb = 0
start = time.time()
cls_weights = np.ones([config.classnum], np.float32)
weights = torch.from_numpy(cls_weights)
weights = weights.cuda()
if __name__ == '__main__':
    for epoch in range(config.epoch_start,config.n_epochs):
        seg_loss_t = 0
        kd_loss_t = 0
        val_Loss = 0
        score = 0
        conf_mat_val = 0
        conf_mat_tra = 0
        loop = tqdm(enumerate(traindata), total = len(traindata))
        for i,data in loop:
            rgbn,sar,m,seg = data            
            rgbn = Variable(rgbn).cuda()
            sar = Variable(sar).cuda()
            m = Variable(m).cuda()
            seg = Variable(seg).cuda()
            
            optimizer.zero_grad()

            if config.amp:
                with autocast():
                    rgbresult = net(rgbn) 
                    ce = CE_Loss(rgbresult,seg)
                    dice = Dice_loss(rgbresult,seg)
                    loss = ce + dice
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                rgbresult = net(rgbn) 

                ce = CE_Loss(rgbresult,seg)
                dice = Dice_loss(rgbresult,seg)

                loss = ce + dice
                loss.backward()
                optimizer.step()
            
            seg_loss_t = seg_loss_t + ce + dice
            
            loop.set_description(f'Epoch [{epoch}/{config.n_epochs}]')
            loop.set_postfix(seg_loss=seg_loss_t.item()/(i+1),lr=optimizer.param_groups[0]['lr'])

        scheduler.step()
        net.eval()
        with torch.no_grad():
            for j,data in enumerate(valdata):
                rgbn,sar,masks,seg = data
                n, c, h, w = seg.size()
                
                rgbn = Variable(rgbn).cuda()
                sar = Variable(sar).cuda()
                masks = Variable(masks).cuda()
                seg = Variable(seg).cuda()
                
                rgb_sar = torch.cat([rgbn,sar],dim=1)
                if config.amp:
                    with autocast():
                        rgbresult = net(rgbn) 
                else:
                    
                    rgbresult = net(rgbn)
                _, preds = torch.max(rgbresult, 1)
                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            
                conf_mat_val += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=config.classnum)

        acc,miou,IoU = metric.evaluate(conf_mat_val)
        print('Val: acc: %.3f, mIoU: %.3f'%(acc,miou))

        f = open(os.path.join(config.save_root,'result.txt'),'a+')
        f.writelines('Epoch:%d,mIoU:%.3f,IoU:%s'%(epoch,miou,str(IoU.tolist()))+'\n')
        f.close()
        if miou-global_Fb > 0.0005:
            print('get better performance from %.3f to %.3f), saving model...'%(global_Fb,miou))
            global_Fb = miou
            torch.save(net.state_dict(),os.path.join(config.save_root,'epoch%d_%.3f.pth'%(epoch,miou)))
