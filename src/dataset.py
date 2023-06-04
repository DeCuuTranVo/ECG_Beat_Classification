import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CustomBeatDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, target_transform=None):       
        self.signals = df.iloc[:, 0:187].to_numpy()
        self.labels = df.iloc[:, 187].to_numpy()
        
        self.signals = torch.from_numpy(self.signals)
        self.signals = torch.unsqueeze(self.signals, 1)
        self.labels = torch.from_numpy(self.labels)
        
        self.signals = self.signals.float()
        self.labels = self.labels.long()
 
        # print(self.signals.shape)
        # print(self.labels.shape)
        
        # signal_size = [187]
        # signal_transformation = [
        #     transforms.ToTensor(),
        #     transforms.ConvertImageDtype(torch.float),
        #     ]
        # transform = transforms.Compose(signal_transformation)
        
        # label_transformation = [
        #     transforms.ToTensor(),
        #     transforms.ConvertImageDtype(torch.float)
        # ]
        # target_transform = transforms.Compose(label_transformation)
        
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self. , self.img_labels.iloc[idx, 0])
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.transform:
            signal = self.transform(signal)
        if self.target_transform:
            label = self.target_transform(label)

        return signal, label
