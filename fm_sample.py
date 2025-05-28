import argparse
import torch
from unet_model import UNetModel
from flow_matching import (
    sample_and_visualize_cfg,
    sample_and_visualize_cfg2,
    find_latest_checkpoint
)

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Load UNet model and visualize samples")
    parser.add_argument("--folder", type=str, default="/root/autodl-tmp", help="Directory where checkpoints are saved")
    parser.add_argument("--model_name", type=str, default="unet_fm", help="Model name prefix for checkpoint")
    parser.add_argument("--model_channels", type=int, default=128, help="model_channels")
    parser.add_argument("--num_classes", type=int, default=200, help="Number of classes (default: TinyImageNet 200)")
    parser.add_argument("--class_ids", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5], help="Class IDs to sample")
    parser.add_argument("--cifar10", action="store_true", help="Use CIFAR-10 settings (num_classes=10, 32x32 resolution)")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动处理 CIFAR-10 情况
    if args.cifar10:
        print("📦 使用 CIFAR-10 模型结构与采样设置")
        args.num_classes = 10
        sample_fn = sample_and_visualize_cfg2
    else:
        print("📦 使用 Tiny-ImageNet 模型结构与采样设置")
        sample_fn = sample_and_visualize_cfg

    # 初始化模型
    model = UNetModel(model_channels=args.model_channels,num_classes=args.num_classes).to(device)

    # 加载模型
    ckpt_path, epoch = find_latest_checkpoint(args.folder, f"{args.model_name}_epoch")
    print(f"🔍 检查点路径：{ckpt_path}")
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ 加载模型成功：{ckpt_path}")

        # 调用对应采样函数
        sample_fn(
            model=model,
            device=device,
            epoch=epoch,
            class_ids=args.class_ids,
            num_per_class=6,
            guidance_scale=2.0,
            save_dir=args.folder,
            model_name=args.model_name,
            max_step=1000
        )
    else:
        print("❌ 未找到模型权重文件，请先训练。")

if __name__ == "__main__":
    main()
