import torch
import matplotlib.pyplot as plt
from model import GaussianDiffusion  # 假设扩散模型类定义在model.py中
from model import UNetModel  # 假设UNet模型定义在此文件中

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和扩散过程
model = UNetModel().to(device)  # 需要与训练时相同的模型结构
diffusion = GaussianDiffusion(timesteps=1000)

# 加载训练好的权重
model.load_state_dict(torch.load("diffusion_model_epoch10.pth", map_location=device))
model.eval()  # 设置为评估模式

# 生成图像函数
def generate_samples(num_samples=16):
    with torch.no_grad():
        # 生成形状为 (num_samples, 1, 28, 28) 的MNIST图像
        samples = diffusion.sample(
            model, 
            image_size=28, 
            batch_size=num_samples, 
            channels=1
        )
        
        # 将输出从 [-1, 1] 转换回 [0, 1]
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        return samples.cpu()

# 生成并可视化结果
generated_images = generate_samples(16)

# 可视化函数
def plot_images(images, n_rows=4):
    n_cols = len(images) // n_rows
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")  # 压缩通道维度
        plt.axis("off")
    plt.show()

# 显示生成结果
plot_images(generated_images)