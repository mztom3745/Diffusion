import argparse
import torch
from model import UNetModel  
from flow_matching import get_dataloader, train 

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train UNetModel with Flow Matching")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/zsz", help="Directory to save checkpoints")
    parser.add_argument("--model_name", type=str, default="unet_fm", help="Model name for checkpointing")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--data_path", type=str, default="/root/autodl-pub/cifar-10", help="Directory to data")

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型和数据加载器
    model = UNetModel().to(device)
    dataloader = get_dataloader(batch_size=args.batch_size,data_path=args.data_path)

    # 训练模型
    train(
        model,
        dataloader,
        device,
        save_dir=args.save_dir,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()
