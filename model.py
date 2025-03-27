# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
import torch
import math
from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
    
# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
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
        model_channels=128,      # 模型基础通道数
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
            # 拼接当前特征与对应的下采样特征（从hs末尾弹出）
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)  # 处理拼接后的特征
        
        # 5. 输出层
        return self.out(h)
    
# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        # 初始化扩散模型的参数
        self.timesteps = timesteps  # 扩散总步数
        
        # 根据指定类型生成 beta 序列
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)  # 线性增长的 beta 序列
        # elif beta_schedule == 'cosine':
        #     betas = cosine_beta_schedule(timesteps)  # 余弦变化的 beta 序列（未实现）
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas  # 形状为 (timesteps,) 的 beta 序列
        
        # 计算 alpha 相关参数
        self.alphas = 1. - self.betas  # alpha_t = 1 - beta_t
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # ᾱ_t = 累乘 alpha_i (i=1~t)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)  # ᾱ_{t-1}，首项填充 1
        
        # 预计算扩散过程参数（用于快速计算）
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # √ᾱ_t（用于前向加噪）
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # √(1-ᾱ_t)（前向加噪）
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)  # log(1-ᾱ_t)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)  # 1/√ᾱ_t（用于反向去噪）
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)  # √(1/ᾱ_t - 1)
        
        # 计算后验分布 q(x_{t-1}|x_t, x_0) 的参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # 后验方差 σ_t^2 = β_t*(1-ᾱ_{t-1})/(1-ᾱ_t)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))  # 对数方差（数值稳定）
        
        # 计算后验均值的系数
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # 系数1：β_t*√ᾱ_{t-1}/(1-ᾱ_t)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )  # 系数2：(1-ᾱ_{t-1})*√α_t/(1-ᾱ_t)
    
    def _extract(self, a, t, x_shape):
        """从张量 a 中提取对应时间步 t 的值，并 reshape 为输入 x 的形状"""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()  # 根据 t 的索引从 a 中取值
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # reshape 为 (batch_size, 1, 1, ...)
        return out  # 使输出与输入 x 的维度匹配（便于广播）
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：根据 x_0 和 t 生成 x_t"""
        if noise is None:
            noise = torch.randn_like(x_start)  # 生成与 x_start 同形的标准正态噪声
        
        # 计算 x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(self, x_start, t):
        """计算 q(x_t|x_0) 的均值、方差和对数方差"""
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start  # μ_t = √ᾱ_t * x_0
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)  # σ_t^2 = 1 - ᾱ_t
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)  # log(σ_t^2)
        return mean, variance, log_variance
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """计算后验分布 q(x_{t-1}|x_t, x_0) 的均值、方差和对数方差"""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start  # 系数1 * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t  # + 系数2 * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)  # σ_t^2
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从 x_t 和预测的噪声 ε 中重建 x_0"""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t  # x_t / √ᾱ_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise  # - √(1/ᾱ_t - 1) * ε
        )  # 结果等于 x_0
    
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        """计算反向过程 p(x_{t-1}|x_t) 的均值、方差"""
        # 用模型预测噪声 ε_θ
        pred_noise = model(x_t, t)
        # 用预测噪声重建 x_0
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)  # 将像素值限制在 [-1, 1]
        
        # 计算 q(x_{t-1}|x_t, x_0) 的参数（此时 x_0 是重建值）
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        """从 p(x_{t-1}|x_t) 中采样 x_{t-1}"""
        # 预测均值和对数方差
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised)
        noise = torch.randn_like(x_t)  # 生成标准正态噪声
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  # t=0 时 mask 为 0（不加噪声）
        # 计算 x_{t-1} = 均值 + 标准差 * 噪声
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """反向扩散循环：从纯噪声逐步生成图像"""
        batch_size = shape[0]
        device = next(model.parameters()).device  # 获取模型所在的设备
        img = torch.randn(shape, device=device)  # 初始化为标准正态分布
        imgs = []
        # 从 T 到 0 逐步去噪
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs  # 返回所有中间步骤结果（可选）
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        """生成图像入口函数"""
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    
    def train_losses(self, model, x_start, t):
        """计算训练损失（预测噪声的 MSE）"""
        noise = torch.randn_like(x_start)  # 生成真实噪声
        x_noisy = self.q_sample(x_start, t, noise=noise)  # 前向加噪得到 x_t
        predicted_noise = model(x_noisy, t)  # 模型预测噪声 ε_θ
        loss = F.mse_loss(noise, predicted_noise)  # 计算均方误差
        return loss