import argparse
import torch
from unet_model import UNetModel  # 你自己写的模型
from flow_matching import get_cifar10_dataloader, train

def main():
    parser = argparse.ArgumentParser(description="Train UNetModel on CIFAR-10 with Flow Matching")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp", help="Directory to save checkpoints")
    parser.add_argument("--data_path", type=str, default="/root/autodl-pub/cifar-10", help="Path to CIFAR-10 dataset")
    parser.add_argument("--model_name", type=str, default="unet_fm", help="Model name for checkpointing")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_step", type=int, default=1000, help="Max diffusion steps")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume training")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📦 使用 CIFAR-10 数据集训练")

    model = UNetModel(num_classes=10).to(device)
    dataloader = get_cifar10_dataloader(batch_size=args.batch_size, data_root=args.data_path)

    train(
        model=model,
        dataloader=dataloader,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        model_name=args.model_name,
        max_step=args.max_step,
        resume_path=args.resume_path  # ✅ 新增
    )

if __name__ == "__main__":
    main()
