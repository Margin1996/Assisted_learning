import torch
import torch.nn.functional as F
import torch.nn as nn
from config import config

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    # inputs B, C, H, W, and target B, H, W, C. 
    # There are C dimensions in total, each dimension representing a class.
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
    #--------------------------------------------#
    #   dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs, axis=[0,1]) - tp
    fn = torch.sum(temp_target, axis=[0,1]) - tp
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def CE_Loss(inputs, target, reduction='mean'):
    # The shape of the input for "CrossEntropyLoss" is N,C, target is N
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1, c)
    temp_target = torch.argmax(temp_target, dim=1).view(-1)
    CE_loss  = nn.CrossEntropyLoss(reduction=reduction)(temp_inputs, temp_target)
    return CE_loss

def KDMask(teacher,student,hard_label):
    with torch.no_grad():
        loss_teacher = F.cross_entropy(teacher, hard_label, reduction='none')
        loss_student = F.cross_entropy(student, hard_label, reduction='none')
        gama = F.relu(loss_student-loss_teacher)  
        #The areas with larger losses in the target branch need learning.

        result_teacher = torch.argmax(torch.softmax(teacher,dim=1),dim=1)
        equal_mask = result_teacher.eq(hard_label).int()

        result_student = torch.argmax(torch.softmax(student,dim=1),dim=1)
        equal_mask_1 = result_student.ne(hard_label).int()
        # Learn only when the student's output differs from the label, 
        # and when the teacher's output is the same as the label.
        mask = equal_mask_1*equal_mask*gama
    return mask

# local knowledge distillation loss
def local_kd_loss(teacher, student, hard_label, temperature=5):
    eps = 1e-6
    kd_mask = KDMask(teacher,student,hard_label)
    soft_student = F.softmax(student / temperature,dim=1)
    soft_teacher = F.softmax(teacher / temperature,dim=1)
    kd_loss = torch.sum(soft_teacher * torch.log(soft_teacher/soft_student + eps),dim=1)
    kd_loss = torch.sum(kd_loss*kd_mask)/torch.sum(kd_mask)
    return kd_loss

# global knowledge distillation loss
def global_kd_loss(teacher, student, hard_label, num_cls = config.classnum):
    kd_loss = 0.0
    temperature = 2
    eps = 1e-6
    result_teacher = torch.argmax(torch.softmax(teacher,dim=1),dim=1)
    equal_mask = result_teacher.eq(hard_label).int()
    result_student = torch.argmax(torch.softmax(student,dim=1),dim=1)
    equal_mask_1 = result_student.ne(hard_label).int()
    mask= hard_label*equal_mask*equal_mask_1

    for i in range(0, num_cls):
        mask_index = (mask == i).int().unsqueeze(1)
        t_logits_mask_out = teacher * mask_index
        t_logits_avg = torch.sum(t_logits_mask_out,dim=[2,3])/(torch.sum(mask_index,dim=[2,3])+eps) #before scaling
        t_soft_prob =F.softmax(t_logits_avg/temperature,dim=1) #after scaling

        s_logits_mask_out = student * mask_index
        s_logits_avg = torch.sum(s_logits_mask_out,dim=[2,3])/(torch.sum(mask_index,dim=[2,3])+eps) #before scaling
        
        s_soft_prob =F.softmax(s_logits_avg/temperature,dim=1) #after scaling

        ## KL divergence loss
        loss = torch.sum(t_soft_prob * torch.log(t_soft_prob/s_soft_prob + eps))
        
        # # ## Cross entrophy
        # s_soft_prob_cls = torch.argmax(s_soft_prob,dim=1)
        # loss = F.cross_entropy(t_soft_prob,s_soft_prob_cls)

        # ## L1 Norm
        # loss = F.l1_loss(t_soft_prob,s_soft_prob)

        kd_loss += loss
    kd_loss = kd_loss / num_cls

    return kd_loss