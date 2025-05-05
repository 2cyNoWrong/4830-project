import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T

class DepthDataset(Dataset):
    def __init__(self, rgb_paths, depth_paths, transform=None):
        self.rgb_paths, self.depth_paths = rgb_paths, depth_paths

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Resize((96, 640)),
            T.Normalize([0.5]*3, [0.5]*3)  # 适用于 RGB
        ])
        self.depth_transform = T.Compose([
            T.Resize((96, 640)),
            T.ToTensor(),                 # 转为 float32
        ])


    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        img = Image.open(self.rgb_paths[idx]).convert("RGB")

        depth = Image.open(self.depth_paths[idx])


        img = self.img_transform(img)  # [3, H, W], float32 in [0, 1]
        depth = self.depth_transform(depth)  # [1, H, W], float32 in [0, 1]
        # print("path:",self.rgb_paths[idx])
        # print("img.shape,depth.shape:",img.shape,depth.shape)
        return img, depth

def get_dataloaders(
    rgb_paths,
    depth_paths,
    transform=None,
    batch_size=4,
    val_ratio=0.1,
    test_ratio=0.1,
    num_workers=4,
    pin_memory=True
):
    dataset = DepthDataset(rgb_paths, depth_paths, transform)
    total = len(dataset)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

def collect_rgb_depth_pairs(rgb_root, depth_root):
    """
    遍历 raw_rgb 和 label 文件夹结构，匹配同名 PNG 文件对
    返回两个路径列表：rgb_paths, depth_paths
    """
    rgb_paths, depth_paths = [], []
    scenes = sorted(os.listdir(depth_root))

    for scene in scenes:
        rgb_dir = os.path.join(rgb_root, scene)
        depth_dir = os.path.join(depth_root, scene)
        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
            continue

        for d_path in sorted(glob(os.path.join(depth_dir, "*.png"))):
            fname = os.path.basename(d_path)
            r_path = os.path.join(rgb_dir, fname)
            if os.path.exists(r_path):
                rgb_paths.append(r_path)
                depth_paths.append(d_path)

    return rgb_paths, depth_paths
