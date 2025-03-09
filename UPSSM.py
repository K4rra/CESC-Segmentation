import torch
import torch.nn as nn
from einops import rearrange
import torch.Function as F

class PSSM(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * dim)

        # 参数投影层
        self.proj = nn.Linear(dim, self.d_inner * 3, bias=False)

        # 状态空间参数
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))

        # 卷积预处理
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

    def forward(self, x):
        """输入形状: (B, L, D)"""
        B, L, D = x.shape

        # 参数投影
        x_proj = self.proj(x)
        x_conv, delta, BC = x_proj.split([self.d_inner, self.d_inner, self.d_inner], dim=-1)

        # 卷积预处理
        x_conv = rearrange(x_conv, 'b l d -> b d l')
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = rearrange(x_conv, 'b d l -> b l d')

        # 离散化参数
        delta = torch.sigmoid(delta)  # Δ ∈ (0,1)
        BC = rearrange(BC, 'b l (n d) -> b l n d', n=2)
        B, C = BC.unbind(dim=2)

        # 离散化状态空间参数
        A_bar = torch.exp(torch.einsum('bnl,ij->bnil', delta, self.A))
        B_bar = torch.einsum('bnl,bnl->bnl', delta, B)

        # 双向扫描
        def scan(x, A, B, reverse=False):
            if reverse:
                x = torch.flip(x, [1])

            h = torch.zeros(B, self.d_state, device=x.device)
            states = []
            for t in range(L):
                h = A[:, t] * h + B[:, t] * x[:, t]
                states.append(h)
            states = torch.stack(states, dim=1)

            if reverse:
                states = torch.flip(states, [1])
            return states

        # 正向扫描
        h_fw = scan(x_conv, A_bar, B_bar)
        # 反向扫描
        h_bw = scan(x_conv, A_bar, B_bar, reverse=True)

        # 合并双向状态
        h = h_fw + h_bw
        y = torch.einsum('bln,bnld->bld', h, C)

        # 残差连接
        y = y + self.D * x_conv
        return y


class VisionMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = UPSSM(dim, d_state=d_state)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim))

    def forward(self, x):
        """输入形状: (B, H, W, C)"""
        residual = x
        x = self.norm(x)

        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b (h w) c')

        # 双向状态空间模型
        x = self.ssm(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)

        # MLP
        x = self.mlp(x)
        return x + residual


class UPSSM(nn.Module):
    def __init__(self, dim, d_state=16, conv_kernel=4):
        """
        Args:
            dim: 输入通道维度
            d_state: 状态空间维度
            conv_kernel: 卷积核大小
        """
        super().__init__()

        # 双流特征提取分支
        self.branch1 = nn.Sequential(
            PSSM(dim, d_state=d_state, d_conv=conv_kernel),
            nn.LayerNorm(dim)
        )

        self.branch2 = nn.Sequential(
            PSSM(dim, d_state=d_state, d_conv=conv_kernel),
            nn.LayerNorm(dim)
        )

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.GELU()
        )

    def spatial_cosine_sim(self, feat1, feat2):
        """计算空间余弦相似度"""
        # 归一化处理
        feat1_norm = F.normalize(feat1, p=2, dim=-1)  # (B, H, W, C)
        feat2_norm = F.normalize(feat2, p=2, dim=-1)

        # 计算相似度
        sim_matrix = torch.einsum('bhwc,bhwc->bhw', feat1_norm, feat2_norm)
        return sim_matrix.unsqueeze(-1)  # (B, H, W, 1)

    def forward(self, img1, img2):
        """
        输入:
            img1: 图像1, 形状 (B, H, W, C)
            img2: 图像2, 形状 (B, H, W, C)
        输出:
            融合特征图, 形状 (B, H, W, C)
        """
        # 特征提取
        feat1 = self.branch1(img1)  # (B, H, W, C)
        feat2 = self.branch2(img2)

        # 计算动态融合权重
        sim_weights = self.spatial_cosine_sim(feat1, feat2)

        # 通道维度重组
        feat1 = rearrange(feat1, 'b h w c -> b c h w')
        feat2 = rearrange(feat2, 'b h w c -> b c h w')
        sim_weights = rearrange(sim_weights, 'b h w 1 -> b 1 h w')

        # 加权特征融合
        fused = sim_weights * feat1 + (1 - sim_weights) * feat2

        # 多尺度特征增强
        fused = torch.cat([feat1, fused], dim=1)  # (B, 2C, H, W)
        fused = self.fusion(fused)  # (B, C, H, W)

        # 恢复原始维度顺序
        fused = rearrange(fused, 'b c h w -> b h w c')
        return fused





#