import argparse
import torch
from unet_model import UNetModel  # 你自己写的模型
from flowmatching import get_dataloader, train_ot

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="Train UNetModel with Flow Matching OT")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/zsz", help="Directory to save checkpoints")
    parser.add_argument("--model_name", type=str, default="unet_fm_ot", help="Model name for checkpointing")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = UNetModel(
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ).to(device)

    # 加载数据
    dataloader = get_dataloader(batch_size=args.batch_size)

    # 开始训练
    train_ot(
        model,
        dataloader,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()
