import torch
import torch.nn as nn
import torch.nn.functional as F

from fno_co2.models.blocks import ResBlock


class FiLMModulation(nn.Module):
    """Modulación afín: y·(1+γ(cond)) + β(cond).
    Los submódulos `.gamma` y `.beta` son nn.Linear para que `build_param_groups`
    los detecte como `no_decay` (convención del baseline)."""

    def __init__(self, cond_dim: int, c: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, c)
        self.beta = nn.Linear(cond_dim, c)

    def forward(self, x, cond_emb):
        """x: (B*T, C, H, W), cond_emb: (B*T, cond_dim)."""
        g = self.gamma(cond_emb).view(-1, x.size(1), 1, 1)
        b = self.beta(cond_emb).view(-1, x.size(1), 1, 1)
        return x * (1.0 + g) + b


class DownBlock(nn.Module):
    """Descenso de resolución: ResBlock + stride-2 convolution."""

    def __init__(self, c_in: int, c_out: int, dropout_p: float = 0.0, use_group_norm: bool = False):
        super().__init__()
        self.res = ResBlock(c_in, dropout_p=dropout_p, use_group_norm=use_group_norm)
        self.down = nn.Conv2d(c_in, c_out, 3, 2, 1, padding_mode="replicate")

    def forward(self, x):
        x = self.res(x)
        x = self.down(x)
        return x


class UpBlock(nn.Module):
    """Ascenso de resolución con skip connection y FiLM."""

    def __init__(
        self,
        c_in: int,
        c_skip: int,
        c_out: int,
        cond_dim: int,
        dropout_p: float = 0.0,
        use_group_norm: bool = False,
    ):
        super().__init__()
        self.res = ResBlock(c_in + c_skip, dropout_p=dropout_p, use_group_norm=use_group_norm)
        self.film = FiLMModulation(cond_dim, c_in + c_skip)
        self.out_conv = nn.Conv2d(c_in + c_skip, c_out, 1)

    def forward(self, x, skip, cond_emb, skip_spatial_size):
        """x: (B*T, C_in, H, W), skip: (B*T, C_skip, H_skip, W_skip), cond_emb: (B*T, cond_dim)."""
        x = F.interpolate(x, size=skip_spatial_size, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        x = self.film(x, cond_emb)
        x = self.out_conv(x)
        return x


class UNetFiLMTemporal(nn.Module):
    """U-Net con condicionamiento temporal FiLM (spec-002).

    Encoder estático (una sola vez) → condicionamiento temporal (reutilizado del baseline)
    → decoder temporal con FiLM por timestep.
    """

    def __init__(
        self,
        time_steps: int = 61,
        in_c: int = 5,
        h_dim: int = 128,
        cond_dim: int = 128,
        dropout_p: float = 0.1,
        use_group_norm: bool = False,
        unet_depth: int = 3,
    ):
        super().__init__()
        self.time_steps = int(time_steps)
        self.unet_depth = unet_depth
        self.cond_dim = cond_dim

        c0 = h_dim
        channels = [c0 * (2**i) for i in range(unet_depth + 1)]

        self.stem = nn.Conv2d(in_c, channels[0], 3, 1, 1, padding_mode="replicate")

        self.down_blocks = nn.ModuleList()
        for i in range(unet_depth):
            self.down_blocks.append(
                DownBlock(
                    channels[i],
                    channels[i + 1],
                    dropout_p=dropout_p,
                    use_group_norm=use_group_norm,
                )
            )

        self.up_blocks = nn.ModuleList()
        for i in range(unet_depth, 0, -1):
            self.up_blocks.append(
                UpBlock(
                    channels[i],
                    channels[i - 1],
                    channels[i - 1],
                    cond_dim,
                    dropout_p=dropout_p,
                    use_group_norm=use_group_norm,
                )
            )

        self.head = nn.Conv2d(channels[0], 2, 3, 1, 1, padding_mode="replicate")

        self.t_embed = nn.Embedding(self.time_steps, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, x, d, inj):
        """
        x: (B, 4, H, W) — propiedades estáticas
        d: (B, 1) — profundidad normalizada
        inj: (B, T, 2) — series de inyección
        """
        b, _, h, w = x.shape

        depth_map = d.view(b, 1, 1, 1).expand(b, 1, h, w)
        z = self.stem(torch.cat([x, depth_map], dim=1))

        skips = []
        skip_shapes = []

        for down in self.down_blocks:
            skips.append(z)
            skip_shapes.append((z.size(2), z.size(3)))
            z = down(z)

        z_bottleneck = z

        if inj.ndim == 2:
            inj = inj.unsqueeze(0)
        if inj.size(1) < self.time_steps:
            pad = torch.zeros(
                b, self.time_steps - inj.size(1), inj.size(2), device=inj.device, dtype=inj.dtype
            )
            inj = torch.cat([inj, pad], dim=1)
        else:
            inj = inj[:, : self.time_steps]

        t_idx = torch.arange(self.time_steps, device=x.device)
        t_emb = self.t_embed(t_idx).unsqueeze(0).expand(b, self.time_steps, -1)
        depth_seq = d.unsqueeze(1).expand(-1, self.time_steps, -1)
        cond_input = torch.cat([inj, depth_seq], dim=2)
        cond_seq = t_emb + self.cond_mlp(cond_input)

        z_bt = z_bottleneck.unsqueeze(1).expand(b, self.time_steps, -1, z_bottleneck.size(2), z_bottleneck.size(3))
        z_bt = z_bt.reshape(b * self.time_steps, -1, z_bottleneck.size(2), z_bottleneck.size(3))
        cond_bt = cond_seq.reshape(b * self.time_steps, -1)

        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            skip_spatial = skip_shapes[-(i + 1)]
            skip_expanded = skip.unsqueeze(1).expand(b, self.time_steps, -1, -1, -1)
            skip_expanded = skip_expanded.reshape(b * self.time_steps, -1, skip.size(2), skip.size(3))
            z_bt = up(z_bt, skip_expanded, cond_bt, skip_spatial)

        out = self.head(z_bt)
        return out.view(b, self.time_steps, 2, h, w)


def build(cfg) -> nn.Module:
    """Función de registro automático (spec-002 Fase 3).
    Construye UNetFiLMTemporal leyendo campos de Config."""
    return UNetFiLMTemporal(
        time_steps=cfg.time_steps,
        in_c=5,
        h_dim=cfg.hidden_dim,
        cond_dim=128,
        dropout_p=cfg.dropout_p,
        use_group_norm=cfg.use_group_norm,
        unet_depth=getattr(cfg, "unet_depth", 3),
    )
