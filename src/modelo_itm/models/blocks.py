import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(c, c, 3, 1, 1, padding_mode="replicate"),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class FiLMSpectralBlock(nn.Module):
    def __init__(self, c, modes, cond_dim=128):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(c, c, modes, modes, dtype=torch.cfloat) * 0.02)
        self.local = nn.Conv2d(c, c, 1)
        self.gamma = nn.Linear(cond_dim, c)
        self.beta = nn.Linear(cond_dim, c)

    def forward(self, x, cond_emb):
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        mh = min(self.modes, x_ft.size(-2))
        mw = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :mh, :mw] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mh, :mw],
            self.weight[:, :, :mh, :mw],
        )
        spec_x = torch.fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")

        y = F.gelu(spec_x + self.local(x))
        g = self.gamma(cond_emb).view(-1, y.size(1), 1, 1)
        b = self.beta(cond_emb).view(-1, y.size(1), 1, 1)
        return y * (1.0 + g) + b
