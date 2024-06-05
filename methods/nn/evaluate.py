import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from processing_tools import noise, fig
from data import tiffs
from data.datasets import (
    TiffDataset,
    StrategyDataset,
)
import tifffile
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os    
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def evaluate_model(input_dir, weights_path, output_dir, num_splits = 2, \
                   strategy = "X:1", batch_size = 2, data_scaling=100, network="unet", multi_gpu=True):

# Параметры
    input_dir = Path(input_dir)
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)

    batch_size = num_splits

# Директория output
    output_dir.mkdir(exist_ok=True)

    datasets = [TiffDataset(input_dir / f"{j}/*.tif") for j in range(num_splits)]
    ds = StrategyDataset(*datasets, strategy=strategy)

    dl = DataLoader(ds, batch_size, shuffle=False,)

    # Опция 1) архитектура UNet
    if network == "unet":
        from methods.nn.architectures.unet import UNet
        net = UNet(1, 1) # 1 входной канал, 1 выходной канал
        if multi_gpu:
            net = nn.DataParallel(net)

    # Опция 2) архитектура DnCNN
    if network == "dncnn":
        from methods.nn.architectures.dncnn import DnCNN
        net = DnCNN(1) # 1 входной канал, 1 выходной канал
        if multi_gpu:
            net = nn.DataParallel(net)
        
    # Загрузка сети (весов)
    state = torch.load(weights_path)
    net.load_state_dict(state["state_dict"])
    net = net.cuda()

    net.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl)):
            inp, _ = batch     
            inp = inp.cuda() * data_scaling
            out = net(inp)
            # Среднее значение по размеру батча
            out = out.mean(dim=0) / data_scaling
            # Формирование 2D numpy-массива
            out_np = out.detach().cpu().numpy().squeeze()
            out_path = str(output_dir / f"output_{i:05d}.tif")
            tifffile.imwrite(out_path, out_np)
    fig.plot_imgs(
   #input=tifffile.imread("./beton_6_exp_2/0/rec_01100.tif"),
   # target=tifffile.imread("./beton_6_exp_2/1/rec_01100_exp_2.tif"),
    denoised=tifffile.imread("./output/output_00000.tif"),
   # vmin=0,
  #  vmax=0.004,
    width=15,
)
    plt.savefig("images.png", bbox_inches='tight')