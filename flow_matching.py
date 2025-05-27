import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# ==================== 通用函数 ====================

def inverse_normalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(tensor.device)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3,1,1).to(tensor.device)
    return tensor * std + mean

def sample_conditional_pt(x0, x1, t, sigma=0.1):
    mu_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def find_latest_checkpoint(folder,dir="unet_fm_epoch"):
    pattern = re.compile(fr"{re.escape(dir)}(\d+)\.pt")
    max_epoch = -1
    filename = None
    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                filename = f
    if filename:
        return os.path.join(folder, filename), max_epoch
    else:
        return None, -1

# ==================== 数据与模型设置 ====================

def get_dataloader(batch_size=256, data_path='/root/autodl-pub/cifar-10'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
# ==================== 采样函数 ====================

def sample_and_visualize(model, device, epoch, save_dir=None,model_name="unet_fm",max_step=1000):
    model.eval()
    with torch.no_grad():
        def ode_func(t, x):
            t_embed = torch.full((x.size(0),), int(t.item() * max_step), device=x.device)
            return model(x, t_embed)

        # 初始随机噪声
        x0 = torch.randn(16, 3, 32, 32).to(device)
        # 解ODE，仅返回最终状态
        t_span = torch.tensor([0.0, 1.0], device=device)
        traj = odeint(ode_func, x0, t_span, rtol=1e-5, atol=1e-5, method='dopri5')
        x1 = traj[-1]  # t=1 的图像

        # 按 4x4 展示最终生成图像
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i in range(16):
            row, col = divmod(i, 4)
            img = torch.clamp(inverse_normalize(x1[i].cpu()), 0, 1)
            axes[row, col].imshow(img.permute(1, 2, 0))
            axes[row, col].axis('off')
        plt.tight_layout()
        # ✅ 保存图像
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_epoch{epoch}.png")
            plt.savefig(save_path)
            print(f"📸 采样图已保存到: {save_path}")
        plt.show()

# ==================== 训练函数 ====================

def train(model, dataloader, device, save_dir="/content/drive/MyDrive/zsz",
          max_step=1000,
          num_epochs=100,savepoch=10, model_name="unet_fm"):
    ckpt_path, start_epoch = find_latest_checkpoint(save_dir,dir=f"{model_name}_epoch")
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ 加载模型成功：{ckpt_path}")
    else:
        print("🚀 未找到训练权重，将从头开始训练")
        start_epoch = 0
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        for batch_idx, (x1, _) in enumerate(dataloader):
            x1 = x1.to(device)
            x0 = torch.randn_like(x1).to(device)

            t_continuous = torch.rand(x1.size(0), device=device)
            t_discrete = (t_continuous * max_step).long().view(-1)
            t_expand = t_continuous.view(-1, 1, 1, 1)

            xt = sample_conditional_pt(x0, x1, t_expand)
            vt = model(xt, t_discrete)
            ut = x1 - x0

            loss = torch.mean((vt - ut) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        if epoch % savepoch == 0 or epoch == num_epochs:
            save_path = os.path.join(save_dir, f"{model_name}_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型第 {epoch} 轮已保存：{save_path}")
            sample_and_visualize(model, device, epoch,save_dir=save_dir,model_name=model_name)
            print(f"✅ 模型第 {epoch} 轮图片已保存：{model_name}_epoch{epoch}.png")

from torch.optim.lr_scheduler import LambdaLR

# 学习率调度器（Polynomial decay with warmup）
def get_polynomial_scheduler(optimizer, warmup_steps, total_steps, power=1.0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, ((total_steps - current_step) / float(max(1, total_steps - warmup_steps))) ** power)
    return LambdaLR(optimizer, lr_lambda)

# 条件采样路径和真实速度
def sample_conditional_pt_2(x0, x1, t, sigma=0.05):
    mu_t = t.view(-1, 1, 1, 1) * x1
    epsilon = torch.randn_like(x0)
    sigma = 1 - (1 - sigma) * t.view(-1, 1, 1, 1)
    return mu_t + sigma * epsilon

def sample_conditional_ut(x0, x1, sigma=0.05):
    return x1 - (1 - sigma) * x0

# 主训练函数
def train_ot(model, dataloader, device, save_dir="/content/drive/MyDrive/zsz",
          max_step=1000, num_epochs=100, savepoch=10,model_name="unet_fm_ot",
          base_lr=5e-4):

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.1)
    print(f"Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")
              
    ckpt_path, start_epoch = find_latest_checkpoint(save_dir,dir=f"{model_name}_epoch")
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ 加载模型成功：{ckpt_path}")
    else:
        print("🚀 未找到训练权重，将从头开始训练")
        start_epoch = 0
    # =重新计算追踪全局步数
    steps_per_epoch = len(dataloader)
    step = start_epoch * steps_per_epoch
    scheduler = get_polynomial_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, power=1.0)
    scheduler.last_epoch = step - 1  # 重要！

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        for batch_idx, (x1, _) in enumerate(dataloader):
            x1 = x1.to(device)
            x0 = torch.randn_like(x1).to(device)

            t_continuous = torch.rand(x1.size(0), device=device)
            t_discrete = (t_continuous * max_step).long().view(-1)
            t_expand = t_continuous.view(-1, 1, 1, 1)

            xt = sample_conditional_pt_2(x0, x1, t_expand)
            vt = model(xt, t_discrete)
            ut = sample_conditional_ut(x0, x1)

            loss = torch.mean((vt - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

        if epoch % savepoch == 0 or epoch == num_epochs:
            save_path = os.path.join(save_dir, f"{model_name}_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型第 {epoch} 轮已保存：{save_path}")
            sample_and_visualize(model, device, epoch,save_dir=save_dir,model_name=model_name)
            print(f"✅ 模型第 {epoch} 轮图片已保存：{model_name}_epoch{epoch}.png")
