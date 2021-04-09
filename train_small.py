# vit train.
import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# from linformer import Linformer
# from vit_pytorch.efficient import ViT
from backbones.vit_pytorch.vit import ViT
torch.backends.cudnn.benchmark = True

print(f"Torch: {torch.__version__}")

# Training settings
batch_size = 1024
epochs = 200
lr = 3e-5
gamma = 0.7
seed = 42
device = 'cuda'


# Load Datasets
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_data():
    train_dir = '../data/train'
    test_dir = '../data/test'
    train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    print(f"Train Data: {len(train_list)}, Test Data: {len(test_list)}")

    labels = [path.split('/')[-1].split('.')[0] for path in train_list]

    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=seed)

    # Augumentation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_data = CatsDogsDataset(train_list, transform=train_transforms)
    valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
    test_data = CatsDogsDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=3)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=3)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=3)

    print(f"Train Data: {len(train_list)}, Validation Data: {len(valid_list)}, Test Data: {len(test_list)}, "
          f"train_loader:{len(train_loader)}, valid_loader:{len(valid_loader)}")

    return train_loader, valid_loader, test_loader


def main_1():
    seed_everything(seed)
    train_loader, valid_loader, test_loader = get_data()

    # model
    # efficient_transformer = Linformer(
    #     dim=128,
    #     seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    #     depth=12, heads=8, k=64
    # )
    # model = ViT(
    #     dim=128, image_size=224,
    #     patch_size=32, num_classes=2, transformer=efficient_transformer, channels=3,
    # ).to(device)
    model = ViT(
        image_size=224,
        patch_size=32, num_classes=2, channels=3,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    # Training
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(f"Epoch: {epoch + 1} - loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss: "
              f"{epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
        if epoch % 10 == 1:
            torch.save(model.state_dict(), f'./checkpoints/pretrained-net-{epoch}.pt')


def main_2():
    from backbones.vit_pytorch.cvt import CvT

    v = CvT(
        num_classes=1000,
        s1_emb_dim=64,  # stage 1 - dimension
        s1_emb_kernel=7,  # stage 1 - conv kernel
        s1_emb_stride=4,  # stage 1 - conv stride
        s1_proj_kernel=3,  # stage 1 - attention ds-conv kernel size
        s1_kv_proj_stride=2,  # stage 1 - attention key / value projection stride
        s1_heads=1,  # stage 1 - heads
        s1_depth=1,  # stage 1 - depth
        s1_mlp_mult=4,  # stage 1 - feedforward expansion factor
        s2_emb_dim=192,  # stage 2 - (same as above)
        s2_emb_kernel=3,
        s2_emb_stride=2,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=3,
        s2_depth=2,
        s2_mlp_mult=4,
        s3_emb_dim=384,  # stage 3 - (same as above)
        s3_emb_kernel=3,
        s3_emb_stride=2,
        s3_proj_kernel=3,
        s3_kv_proj_stride=2,
        s3_heads=4,
        s3_depth=10,
        s3_mlp_mult=4,
        dropout=0.
    )

    img = torch.randn(1, 3, 224, 224)

    pred = v(img)  # (1, 1000)
    print(pred)


if __name__ == "__main__":
    main_2()


"""
Dogs vs. Cats Redux: Kernels Edition - https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
data: 20000 train, 5000 valid. 2 class. 
cpu: 7000%, gpu:0,10%. 太慢. 慢在dataloder, mem->gpu
有监督的学习. 
Epoch : 20 - loss : 0.5881 - acc: 0.6843 - val_loss : 0.5891 - val_acc: 0.6806.  不高

lr = 3e-3, bs=1024 不收敛

# mount -t tmpfs -o size=140G  tmpfs /train_tmp
"""
