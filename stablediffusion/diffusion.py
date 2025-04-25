import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb):
        super().__init__()
        self.linear_1 = nn.Linear(time_emb, time_emb * 4)
        self.linear_2 = nn.Linear(time_emb * 4, time_emb * 4)

    def forward(self, x):
        # (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        # (1, 1280)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_times=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_times, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, channels, n_heads=8, d_context=768):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (b, channels, h, w) -> (b, channels, h*2, w*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self, in_channels, interim_dim):
        super().__init__()

        # 各阶段的通道维度
        dim1 = interim_dim  # 320
        dim2 = interim_dim * 2  # 640
        dim3 = interim_dim * 4  # 1280

        self.encoders = nn.ModuleList([
            # Stage 1: (b, 4, h, w) -> (b, 320, h/8, w/8)
            SwitchSequential(nn.Conv2d(in_channels, dim1, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(dim1, dim1), UNET_AttentionBlock(dim1)),
            SwitchSequential(UNET_ResidualBlock(dim1, dim1), UNET_AttentionBlock(dim1)),

            # Stage 2: Downsample -> (b, 320, h/8, w/8) -> (b, 320, h/16, w/16)
            SwitchSequential(nn.Conv2d(dim1, dim1, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(dim1, dim2), UNET_AttentionBlock(dim2)),
            SwitchSequential(UNET_ResidualBlock(dim2, dim2), UNET_AttentionBlock(dim2)),

            # Stage 3: Downsample -> (b, 640, h/16, w/16) -> (b, 1280, h/32, w/32)
            SwitchSequential(nn.Conv2d(dim2, dim2, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(dim2, dim3), UNET_AttentionBlock(dim3)),
            SwitchSequential(UNET_ResidualBlock(dim3, dim3), UNET_AttentionBlock(dim3)),

            # Stage 4: Downsample -> (b, 1280, h/32, w/32) -> (b, 1280, h/64, w/64)
            SwitchSequential(nn.Conv2d(dim3, dim3, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(dim3, dim3)),
            SwitchSequential(UNET_ResidualBlock(dim3, dim3)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(dim3, dim3),
            UNET_AttentionBlock(dim3),
            UNET_ResidualBlock(dim3, dim3)
        )

        self.decoders = nn.ModuleList([
            # (b, 2560, h/64, w/64) -> (b, 2560, h/64, w/64)
            SwitchSequential(UNET_ResidualBlock(dim3 + dim3, dim3)),
            # (b, 2560, h/64, w/64) -> (b, 2560, h/64, w/64)
            SwitchSequential(UNET_ResidualBlock(dim3 + dim3, dim3)),
            # (b, 2560, h/64, w/64) -> (b, 1280, h/32, w/32)
            SwitchSequential(UNET_ResidualBlock(dim3 + dim3, dim3), UpSample(dim3)),

            # (b, 2560, h/32, w/32) -> (b, 1280, h/32, w/32)
            SwitchSequential(UNET_ResidualBlock(dim3 + dim3, dim3), UNET_AttentionBlock(dim3)),
            # (b, 2560, h/32, w/32) -> (b, 1280, h/32, w/32)
            SwitchSequential(UNET_ResidualBlock(dim3 + dim3, dim3), UNET_AttentionBlock(dim3)),
            # (b, 1920, h/32, w/32) -> (b, 1280, h/16, w/16)
            SwitchSequential(UNET_ResidualBlock(dim3 + dim2, dim3), UNET_AttentionBlock(dim3), UpSample(dim3)),

            # (b, 1920, h/16, w/16) -> (b, 640, h/16, w/16)
            SwitchSequential(UNET_ResidualBlock(dim3 + dim2, dim2), UNET_AttentionBlock(dim2)),
            # (b, 1280, h/16, w/16) -> (b, 640, h/16, w/16)
            SwitchSequential(UNET_ResidualBlock(dim2 + dim2, dim2), UNET_AttentionBlock(dim2)),
            # (b, 960, h/16, w/16) -> (b, 640, h/8, w/8)
            SwitchSequential(UNET_ResidualBlock(dim2 + dim1, dim2), UNET_AttentionBlock(dim2), UpSample(dim2)),

            # (b, 960, h/8, w/8) -> (b, 320, h/8, w/8)
            SwitchSequential(UNET_ResidualBlock(dim2 + dim1, dim1), UNET_AttentionBlock(dim1)),
            # (b, 640, h/8, w/8) -> (b, 320, h/8, w/8)
            SwitchSequential(UNET_ResidualBlock(dim1 + dim1, dim1), UNET_AttentionBlock(dim1)),
            # (b, 640, h/8, w/8) -> (b, 320, h/8, w/8)
            SwitchSequential(UNET_ResidualBlock(dim1 + dim1, dim1), UNET_AttentionBlock(dim1)),
        ])

    def forward(self, x, context, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
        x = self.bottleneck(x, context, time)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)
        return x


class UNET_Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self, in_channels=4, interim_dim=320, out_channels=4):
        super().__init__()
        self.time_embedding = TimeEmbedding(interim_dim)
        self.unet = UNET(in_channels, interim_dim)
        self.final = UNET_Output(interim_dim, out_channels)

    def forward(self, latent, context, time):
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        # (b, 4, h/8, w/8) -> (b, 320, h/8, w/8)
        output = self.unet(latent, context, time)
        # (b, 320, h/8, w/8) -> (b, 4, h/8, w/8)
        output = self.final(output)
        return output
