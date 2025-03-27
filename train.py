import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import GaussianDiffusion  # 假设这是你的扩散模型实现
from model import UNetModel

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
timesteps = 1000  # 扩散步数
batch_size = 128
epochs = 10
learning_rate = 1e-4

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),                # 转换为Tensor [0,1]
    transforms.Lambda(lambda x: (x - 0.5) * 2.0),  # 归一化到 [-1, 1]
    transforms.Lambda(lambda x: x.view(-1, 28, 28))  # 确保图像维度正确
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)



# 初始化模型和扩散过程
model = UNetModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
diffusion = GaussianDiffusion(timesteps=timesteps)  # 假设扩散过程类已实现

# 训练循环
for epoch in range(epochs):
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        images = images.to(device)
        batch_size = images.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        # 计算扩散损失
        loss = diffusion.train_losses(model, images, t)
        
        if step % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        loss.backward()
        optimizer.step()
    
    # 每个epoch保存模型
    torch.save(model.state_dict(), f"diffusion_model_epoch{epoch+1}.pth")