#时间模块定义
import torch
import torch.nn as nn
import torch.optim as optim
import math
from abc import abstractmethod
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)
# Attention block with shortcut

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, target_size=None):
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)

# 定义完整的UNet模型，包含注意力机制和时间步嵌入
class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,          # 输入图像的通道数
        model_channels=256,      # 模型基础通道数
        out_channels=3,         # 输出图像的通道数
        num_res_blocks=2,       # 每个分辨率的残差块数量
        attention_resolutions=(8, 16),  # 使用注意力的分辨率（下采样倍数）
        dropout=0,               # Dropout率
        channel_mult=(1, 2, 2, 2),  # 各层通道数的倍增系数
        conv_resample=True,     # 使用卷积进行上下采样（否则使用最近邻）
        num_heads=4             # 注意力头数
    ):
        super().__init__()

        # 保存参数到类实例
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # 时间嵌入层（将时间步编码为高维向量）
        time_embed_dim = model_channels * 4  # 时间嵌入维度是模型通道的4倍
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),  # 线性投影
            nn.SiLU(),                                  # 激活函数
            nn.Linear(time_embed_dim, time_embed_dim),  # 再次投影
        )

        # 下采样模块构建
        self.down_blocks = nn.ModuleList([
            # 初始卷积层（输入通道→model_channels）
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]  # 记录各层输出通道数（用于跳跃连接）
        ch = model_channels                   # 当前通道数跟踪变量
        ds = 1                               # 当前下采样倍数（初始分辨率）

        # 遍历每个分辨率阶段（由channel_mult决定层数）
        for level, mult in enumerate(channel_mult):
            # 每个分辨率阶段添加num_res_blocks个残差块
            for _ in range(num_res_blocks):
                layers = [
                    # 残差块（输入通道ch，输出通道mult*model_channels）
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels  # 更新当前通道数
                # 如果当前分辨率需要注意力机制
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                # 将层序列加入下采样块
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)  # 记录通道数

            # 如果不是最后一个阶段，添加下采样层
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)  # 下采样不改变通道数
                ds *= 2  # 下采样倍数翻倍（如从1→2，2→4等）

        # 中间模块（包含残差块→注意力→残差块）
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),  # 保持通道数
            AttentionBlock(ch, num_heads=num_heads),          # 注意力机制
            ResidualBlock(ch, ch, time_embed_dim, dropout)    # 保持通道数
        )

        # 上采样模块构建
        self.up_blocks = nn.ModuleList([])
        # 逆序遍历channel_mult（从高层到底层）
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # 每个分辨率阶段处理num_res_blocks+1次（包含跳跃连接）
            for i in range(num_res_blocks + 1):
                layers = [
                    # 残差块（输入通道=当前通道+跳跃连接的通道，输出通道=model_channels*mult）
                    ResidualBlock(
                        ch + down_block_chans.pop(),  # 拼接当前特征与跳跃连接
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult  # 更新当前通道
                # 如果需要在该分辨率添加注意力
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                # 如果是当前阶段的最后一个块且不是最低分辨率，添加上采样
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2  # 上采样后分辨率翻倍（如16→8）
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        # 输出层（归一化→激活→卷积）
        self.out = nn.Sequential(
            norm_layer(ch),  # 归一化层（需根据实际情况定义，如nn.GroupNorm）
            nn.SiLU(),       # 激活函数
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),  # 输出卷积
        )

    def forward(self, x, timesteps):
        """
        前向传播
        :param x: 输入张量 [N x C x H x W]
        :param timesteps: 时间步张量 [N]
        :return: 输出张量 [N x C x H x W]
        """
        hs = []  # 保存各层特征用于跳跃连接

        # 1. 时间步嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # 2. 下采样阶段
        h = x  # 输入
        for module in self.down_blocks:
            h = module(h, emb)  # 每个模块处理（传入特征和时间嵌入）
            hs.append(h)         # 保存特征

        # 3. 中间阶段
        h = self.middle_block(h, emb)

        # 4. 上采样阶段
        for module in self.up_blocks:
          skip = hs.pop()

          # 如果 spatial size 不一致，则 resize 当前特征图 h
          if h.shape[2:] != skip.shape[2:]:
              h = F.interpolate(h, size=skip.shape[2:], mode='nearest')  # 保证拼接维度一致

          cat_in = torch.cat([h, skip], dim=1)
          h = module(cat_in, emb)

        # 5. 输出层
        return self.out(h)
