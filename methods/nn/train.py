import numpy as np
from pathlib import Path
from data.datasets import (
    TiffDataset,
    StrategyDataset,
)
import tifffile
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import os    
import argparse

from pathlib import Path
import numpy as np
import torch
import re
from itertools import combinations
from torch.utils.data import (
    Dataset
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_model(train_dir, output_dir, network="unet", num_splits=2, strategy="X:1", \
                epochs = 100, batch_size=7, data_scaling=100, multi_gpu = True):
    
    train_dir = Path(train_dir)
    output_dir = Path(output_dir)

    datasets = [TiffDataset(train_dir / f"{j}/*.tif") for j in range(num_splits)]
    train_ds = StrategyDataset(*datasets, strategy=strategy)

    #train_ds.num_slices, train_ds.num_splits

    dl = DataLoader(train_ds, batch_size, shuffle=True,)


    # Опция 1) архитектура UNet
    if network == "unet":
        from methods.nn.architectures.unet import UNet
        net = UNet(1, 1).cuda() # 1 входной канал, 1 выходной канал
        if multi_gpu:
            net = nn.DataParallel(net)

        optim = torch.optim.Adam(net.parameters())

    # Опция 2) архитектура DnCNN
    if network == "dncnn":
        from methods.nn.architectures.dncnn import DnCNN
        net = DnCNN(1).cuda() # 1 входной канал, 1 выходной канал
        if multi_gpu:
            net = nn.DataParallel(net)

        optim = torch.optim.Adam(net.parameters())

    output_dir.mkdir(exist_ok=True)

    # Набор данных содержит несколько пар input-target для каждого среза.
    # Поэтому мы делим на количество разбиений, чтобы получить эффективное количество эпох.
    train_epochs = max(epochs // num_splits, 1)

    # цикл обучения
    for epoch in range(train_epochs):
        # обучение
        for (inp, tgt) in tqdm(dl):
            inp = inp.cuda(non_blocking=True) * data_scaling
            tgt = tgt.cuda(non_blocking=True) * data_scaling

            # Шаг обучения
            output = net(inp)
            loss = nn.functional.mse_loss(output, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Сохранение сети (весов)
        torch.save(
            {"epoch": int(epoch), "state_dict": net.state_dict(), "optimizer": optim.state_dict()}, 
            output_dir / f"weights_epoch_{epoch}.torch"
           )
    
        torch.save(
            {"epoch": int(epoch), "state_dict": net.state_dict(), "optimizer": optim.state_dict()}, 
            output_dir / "weights.torch"
)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test args')
    parser.add_argument('-i', '--input', required=True, type=argparse.FileType('r'))

    args = parser.parse_args()

    hyperparams = args.input.readlines()
    train_model(hyperparams)