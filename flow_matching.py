import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os
import re

# ==================== 通用函数 ====================
def inverse_normalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    return tensor * std + mean

def sample_conditional_pt(x0, x1, t, sigma=0.1):
    mu_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def sample_conditional_ut(x0, x1, sigma=0.1):
    return x1 - x0

def get_cifar10_dataloader(batch_size=256, train=True, data_root="./cifar10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = CIFAR10(root=data_root, train=train, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train)

# ==================== 模型定义 ====================
from unet_model import UNetModel

# ==================== 学习率调度器 ====================
def get_polynomial_scheduler(optimizer, warmup_steps, total_steps, power=1.0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, ((total_steps - current_step) / float(max(1, total_steps - warmup_steps))) ** power)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==================== 训练函数 ====================
def train(model, dataloader, device, save_dir="/root/autodl-tmp", max_step=1000, num_epochs=100,
          base_lr=1e-4, savepoch=10, warmup_steps=10, model_name="unet_fm", resume_path=None):

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = get_polynomial_scheduler(optimizer, warmup_steps, total_steps)

    # ====== Resume from checkpoint ======
    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        match = re.search(r"epoch(\d+)", resume_path)
        start_epoch = int(match.group(1)) if match else 0
        print(f"✅ 从 checkpoint 加载模型：{resume_path}，恢复训练从 epoch {start_epoch}")
    else:
        print("🚀 未设置 resume_path 或文件不存在，训练将从头开始")
        start_epoch = 0

    scheduler.last_epoch = start_epoch * steps_per_epoch - 1
    step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        for batch_idx, (x1, labels) in enumerate(dataloader):
            x1, labels = x1.to(device), labels.to(device)
            x0 = torch.randn_like(x1).to(device)

            t_continuous = torch.rand(x1.size(0), device=device)
            t_discrete = (t_continuous * max_step).long().view(-1)
            t_expand = t_continuous.view(-1, 1, 1, 1)

            xt = sample_conditional_pt(x0, x1, t_expand)
            ut = sample_conditional_ut(x0, x1)

            cond = labels.to(device)  # 恒定使用标签

            vt = model(xt, t_discrete, cond=cond)

            loss = torch.mean((vt - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}, Loss: {loss.item():.4f}")

        if epoch % savepoch == 0 or epoch == num_epochs:
            save_path = os.path.join(save_dir, f"{model_name}_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型第 {epoch} 轮已保存：{save_path}")
            sample_and_visualize_cfg(model, device, epoch, class_ids=list(range(10)),
                                      num_per_class=6, guidance_scale=2.5,
                                      save_dir=save_dir, model_name=model_name)

# ==================== CFG 采样 ====================
def sample_and_visualize_cfg(model, device, epoch, class_ids=None, num_per_class=6, guidance_scale=0.0,
                              save_dir=None, model_name="unet_fm", max_step=1000):
    if class_ids is None:
        class_ids = list(range(6))
    model.eval()
    with torch.no_grad():
        all_images = []
        for class_id in class_ids:
            cond = torch.full((num_per_class,), class_id, device=device, dtype=torch.long)
            x = torch.randn(num_per_class, 3, 32, 32, device=device)

            def ode_func(t, x_input):
                t_embed = torch.full((x_input.size(0),), int(t.item() * max_step), device=x.device)
                pred = model(x_input, t_embed, cond=cond)
                return pred

            try:
                t_span = torch.tensor([0.0, 1.0], device=device)
                traj = odeint(ode_func, x, t_span, rtol=1e-5, atol=1e-5, method='dopri5')
                imgs = traj[-1]
                all_images.append(imgs.cpu())
            except Exception as e:
                print(f"❌ 采样类 {class_id} 时失败: {e}")

        if not all_images:
            print("❌ 没有生成任何图像。")
            return

        all_images = torch.cat(all_images, dim=0)
        rows = len(class_ids)
        cols = num_per_class
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i in range(rows * cols):
            row, col = divmod(i, cols)
            ax = axes[row, col]
            if i < len(all_images):
                img = torch.clamp(inverse_normalize(all_images[i]), 0, 1)
                ax.imshow(img.permute(1, 2, 0))
                ax.set_title(f"class {class_ids[row]}", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_cfgsample_epoch{epoch}.png")
            plt.savefig(save_path)
            print(f"📸 CIFAR-10 采样图已保存到: {save_path}")
        plt.show()

# ==================== 运行入口 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_cifar10_dataloader()
    model = UNetModel(in_channels=3, out_channels=3, num_classes=10).to(device)
    train(model, dataloader, device)
