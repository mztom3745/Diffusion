import argparse
import torch
from unet_model import UNetModel  # 你自己写的模型
from flowmatching import sample_and_visualize, find_latest_checkpoint

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Load UNet model and visualize samples")
    parser.add_argument("--folder", type=str, required=True, help="Directory where checkpoints are saved")
    parser.add_argument("--dir", type=str, required=True, help="Subdirectory or pattern prefix for checkpoint files")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = UNetModel().to(device)

    # 查找并加载最新 checkpoint
    ckpt_path, epoch = find_latest_checkpoint(args.folder, args.dir)
    print(f"🔍 检查点路径：{ckpt_path}")
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ 加载模型成功：{ckpt_path}")
        sample_and_visualize(model, device, epoch=epoch, save_dir=args.folder,model_name="unet_fm",max_step=1000)
    else:
        print("❌ 未找到模型权重文件，请先训练。")

if __name__ == "__main__":
    main()
