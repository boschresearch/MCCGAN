import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    """
        Custom Dataset to load image and corresponding target count vector

        Args:
            data (list)                    : list of image samples
            targets (list)                 : list of corresponding count vector for images
            transform (callable, optional) : Optional transform to be applied on a sample
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


def prepareDataLoader(image_array, countvec_path, img_size, batch_size):
    """
        To prepare pytorch dataloader with images and corresponding ground truth count vector

        Parameters:
            image_array (float)     : Image array
            countvec_path (string)  : path to the ground truth count csv file
            img_size (int)          : Image dimension
            batch_size (int)        : Batch size for training
        Returns:
            dataloader              : The configured pytorch dataloder
            count_gt (float)        : The ground truth count vector list
    """
    data = list(np.load(image_array, allow_pickle=True))
    count_gt = list(np.genfromtxt(countvec_path, delimiter=','))
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MyDataset(data, count_gt, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dataloader, count_gt
