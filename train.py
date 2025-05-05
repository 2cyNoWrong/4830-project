# train_and_test.py
# 包含训练脚本 (train) 和 测试脚本 (test)，包含论文中指定的训练策略与增强

import os
import argparse
import torch
import torch.nn.functional as F
# from torch.optim import AdamW

from bitsandbytes.optim import Adam8bit as AdamW

from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
import numpy as np

from torchvision import transforms
from dataset.dataset import get_dataloaders, collect_rgb_depth_pairs
from utils.compute_metrics import compute_metrics
from model.DepthEstimationModel import DepthEstimationModel

# ----------------------
# 数据增强策略
# ----------------------
def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomResizedCrop((375, 1242), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((375, 1242)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

# ----------------------
# 训练函数
# ----------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = DepthEstimationModel(pretrained_sd=args.pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # 使用 collect_rgb_depth_pairs 正确构造全路径列表
    rgb_paths, depth_paths = collect_rgb_depth_pairs("dataset/raw_rgb", "dataset/label")

    # 然后传入 get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        # transform=get_eval_transform(),
        batch_size=1,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=6e-4,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,

    )

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, depths in pbar:
            imgs, depths = imgs.to(device), depths.to(device)



            # 2) 前向计算
            depth_pred, noise, _ = model(imgs)
            # 1) 检查数据中是否有 NaN
            if torch.isnan(depths).any():
                print("Warning: depth labels contain NaN!")
            if torch.isnan(depth_pred).any():
                print("Warning: output contain NaN!")
            # 3) 只在 depth>0 的位置计算 loss
            mask = depths > 0
            valid = mask.sum()
            print(valid)
            # 4) 如果没有任何有效像素，跳过这个 batch
            if valid == 0:
                pbar.set_postfix({'loss': 'skipped'})
                continue

            # 5) 否则正常计算 loss 并优化
            loss = F.l1_loss(depth_pred[mask], depths[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({'loss': loss.item()})

        # 验证阶段
        model.eval()
        metrics = {}
        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs, depths = imgs.to(device), depths.to(device)
                noise_pred, noise, _ = model(imgs)
                preds = noise_pred.squeeze(1)
                mask = depths > 0
                m = compute_metrics(preds, depths.squeeze(1), mask.squeeze(1))
                for k, v in m.items():
                    metrics.setdefault(k, []).append(v)
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        print(f"Validation Metrics Epoch {epoch}: {avg_metrics}")

        torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch{epoch}.pth"))


from torch.cuda.amp import autocast, GradScaler

def train_mix(args):
    """
    Mixed‐precision training with gradient checkpointing and partial freezing of SeeCoder.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # …
    model = DepthEstimationModel(pretrained_sd=args.pretrained).to(device)

    # 开启 UNet 梯度检查点
    if hasattr(model.unet, "enable_gradient_checkpointing"):
        model.unet.enable_gradient_checkpointing()

    # 冻结 SeeCoder Backbone
    for name, p in model.unet.seecoder.backbone.named_parameters():
        p.requires_grad = False

    # 下面不变：只优化 UNet 和 Decoder 以及 SeeCoder 的 decoder/query 部分
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    # …

    scaler = GradScaler()

    # 数据加载
    rgb_paths, depth_paths = collect_rgb_depth_pairs("dataset/raw_rgb", "dataset/label")
    train_loader, val_loader, _ = get_dataloaders(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[MP] Epoch {epoch+1}/{args.epochs}")
        for imgs, depths in pbar:
            imgs, depths = imgs.to(device), depths.to(device)
            optimizer.zero_grad()
            if torch.isnan(depths).any():
                print("标签含 NaN！")

            # 混合精度前向 + 损失计算
            with autocast():
                depth_pred, noise, _ = model(imgs)
                mask = depths > 0
                loss = F.l1_loss(depth_pred[mask], depths[mask])

            # 反向 + 优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        metrics = {}
        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs, depths = imgs.to(device), depths.to(device)
                depth_pred, noise, _ = model(imgs)
                preds = depth_pred.squeeze(1)
                mask = depths > 0
                m = compute_metrics(preds, depths.squeeze(1), mask.squeeze(1))
                for k, v in m.items():
                    metrics.setdefault(k, []).append(v)
        avg = {k: float(np.mean(v)) for k, v in metrics.items()}
        print(f"[Val] Epoch {epoch+1}: {avg}")

        # 保存
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, f"mp_epoch{epoch+1}.pth"))

# ----------------------
# 测试函数
# ----------------------
def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = DepthEstimationModel(pretrained=args.pretrained).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, _, test_loader = get_dataloaders(
        args.rgb_dir,
        args.depth_dir,
        # transform=get_eval_transform(),
        batch_size=args.batch_size,
        val_ratio=0.0,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers
    )

    metrics = {}
    with torch.no_grad():
        for imgs, depths in tqdm(test_loader, desc="Testing"):
            imgs, depths = imgs.to(device), depths.to(device)
            noise_pred, noise, _ = model(imgs)
            preds = noise_pred.squeeze(1)
            mask = depths > 0
            m = compute_metrics(preds, depths.squeeze(1), mask.squeeze(1))
            for k, v in m.items():
                metrics.setdefault(k, []).append(v)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"Test Metrics: {avg_metrics}")

# ----------------------
# 主入口
# ----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'train_mix','test'], required=True)
    parser.add_argument('--pretrained', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--rgb_dir', type=str, required=True)
    parser.add_argument('--depth_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, help='测试时加载的模型路径')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == 'train':
        train(args)
    if args.mode == 'train_mix':
        train_mix(args)
    else:
        test(args)
