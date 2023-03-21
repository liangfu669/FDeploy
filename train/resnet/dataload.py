import torch
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


# 1.create dataset
class MyDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),  # convert PIL.Image to tensor, which is GY
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalization
            ])
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx: int):
        # img to tensor, label to tensor
        img_path = self.path_list[idx]
        abs_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(abs_img_path)
        img = self.transform(img)

        if img_path.split('.')[0] == 'dog':
            label = 1
        else:
            label = 0
        label = torch.as_tensor(label,
                                dtype=torch.int64)  # must use long type, otherwise raise error when training, "expect long"
        return img, label

    def __len__(self) -> int:
        return len(self.path_list)


# 2.dataset split
def dataset_split(full_ds, train_rate):
    train_size = int(len(full_ds) * train_rate)
    validate_size = len(full_ds) - train_size
    train_ds, validate_ds = torch.utils.data.random_split(full_ds, [train_size, validate_size])
    return train_ds, validate_ds


# 3. data loader
def dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return data_loader
