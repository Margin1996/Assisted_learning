import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from loss import Dice_loss,CE_Loss,global_kd_loss,local_kd_loss
from torch.autograd import Variable
from dataset import MyDataset
from config import config
import metric
import time
from torch.cuda.amp import GradScaler as Gradscaler
from torch.cuda.amp import autocast 
from tqdm import tqdm

from hrnet.hrnet import HRnet
from unet import U_Net
scaler = Gradscaler()

traindd = MyDataset(config.trainroot,is_training=True)
traindata = DataLoader(traindd,batch_size=config.batch_size, shuffle=True)
valdata = DataLoader(MyDataset(config.valroot,is_training=False), num_workers=0, batch_size=config.batch_size, shuffle=False)
studentnet = HRnet(in_channel = 1,num_classes=config.classnum,backbone='hrnetv2_w32').cuda() #target modality
teachernet = HRnet(in_channel = 3,num_classes=config.classnum,backbone='hrnetv2_w32').cuda() #auxiliary modality
# teachernet = U_Net(4,config.classnum).cuda() 
teachernet.load_state_dict(torch.load("..\model.pth")) # load the teacher model
teachernet.eval()
optimizer = torch.optim.SGD(studentnet.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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
        l_kd_loss_t = 0
        g_kd_loss = 0
        conf_mat_tra = 0
        conf_mat_val = 0
        loop = tqdm(enumerate(traindata), total = len(traindata))
        for i,data in loop:
            rgbn,sar,m,seg = data
            # traindd.updata_size()
            rgbn = Variable(rgbn).cuda()
            sar = Variable(sar).cuda()
            m = Variable(m).cuda()
            seg = Variable(seg).cuda()

            optimizer.zero_grad()

            if config.amp:
                with autocast():
                    with torch.no_grad():
                        tea_result = teachernet(rgbn)
                    stu_result = studentnet(sar)  
                    ce = CE_Loss(stu_result,seg)
                    dice = Dice_loss(stu_result,seg)
                    lkd = local_kd_loss(tea_result,stu_result,m)
                    gkd = global_kd_loss(tea_result,stu_result,m)
                    loss = ce + dice + gkd + lkd
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    tea_result = teachernet(rgbn)
                stu_result = studentnet(sar)
                ce = CE_Loss(stu_result,seg)
                dice = Dice_loss(stu_result,seg)
                lkd = local_kd_loss(tea_result,stu_result,m)
                gkd = global_kd_loss(tea_result,stu_result,m)
                loss = ce + dice + gkd + lkd
                loss.backward()
                optimizer.step()
            seg_loss_t = seg_loss_t + ce + dice
            l_kd_loss_t = l_kd_loss_t +  lkd.data
            g_kd_loss = g_kd_loss + gkd.data
            
            if torch.isnan(seg_loss_t):
                break
            loop.set_description(f'Epoch [{epoch}/{config.n_epochs}]')
            loop.set_postfix(seg_loss=seg_loss_t.item()/(i+1),local_kd=l_kd_loss_t.item()/(i+1),global_kd=g_kd_loss.item()/(i+1),lr=optimizer.param_groups[0]['lr']) #
        

        scheduler.step()
        studentnet.eval()
        with torch.no_grad():
            for j,data in enumerate(valdata):
                rgbn,sar,masks,seg = data
                n, c, h, w = seg.size()
                
                rgbn = Variable(rgbn).cuda()
                sar = Variable(sar).cuda()
                masks = Variable(masks).cuda()
                seg = Variable(seg).cuda()
                if config.amp:
                    with autocast():
                        stu_result = studentnet(sar)
                else:
                    stu_result = studentnet(sar)

                _, preds = torch.max(stu_result, 1)
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
            torch.save(studentnet.state_dict(),os.path.join(config.save_root,'epoch%d_%.3f.pth'%(epoch,miou)))
        
