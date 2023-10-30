from easydict import EasyDict as ed

config = ed({
    "trainroot": r"data\train", 
    "valroot": r"data\val", 
    "save_root": "model_sar_rgb", 
    "amp": False, #是否使用fp16,损失计算与梯度计算为半精度
    "classnum": 8, #dsm 6 sar 8
    "batch_size": 8, #batch_size
    "lr" : 1e-4, #leanring rate
    "epoch_start": 0,
    "n_epochs": 100,
    "val_time": 5,
    "image_size": 512
})