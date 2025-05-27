from unet_model import UNetModel  #你自己写的模型
from utils import get_dataloader, train_ot
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetModel(
    in_channels=3,
    model_channels=256,
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=(8,16),
    dropout=0.0,
    channel_mult=(1, 2, 2, 2),
    conv_resample=True,
    num_heads=4
).to(device)
dataloader = get_dataloader(batch_size=256)
train_ot(model, dataloader, device='cuda',num_epochs=100,
      save_dir="/content/drive/MyDrive/zsz", model_name='unet_fm_ot')
