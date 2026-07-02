import torch
import torch.nn as nn

from modelo_itm.models.blocks import FiLMSpectralBlock, ResBlock


class PhysicalFNOArchitecture(nn.Module):
    def __init__(self, time_steps=61, in_c=5, h_dim=128, modes=16, cond_dim=128, dropout_p=0.1):
        """in_c: canales totales que entran al encoder DESPUES de concatenar el
        canal de profundidad (ver forward: torch.cat([x, depth_map], dim=1)).
        Con el dataset real (4 propiedades estaticas: AFI/COH/PERM/PORO), x trae
        4 canales y depth_map 1 -> in_c=5 (default) es el total correcto, no un
        "+1" adicional sobre 5."""
        super().__init__()
        self.time_steps = int(time_steps)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, h_dim, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            ResBlock(h_dim, dropout_p=dropout_p),
        )
        self.t_embed = nn.Embedding(self.time_steps, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.fno_blocks = nn.ModuleList([FiLMSpectralBlock(h_dim, modes, cond_dim=cond_dim) for _ in range(4)])
        self.decoder = nn.Sequential(
            ResBlock(h_dim, dropout_p=dropout_p),
            nn.Conv2d(h_dim, h_dim // 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(h_dim // 2, 2, 1),
        )

    def forward(self, x, d, inj):
        b, _, h, w = x.shape
        depth_map = d.view(b, 1, 1, 1).expand(b, 1, h, w)
        z = self.encoder(torch.cat([x, depth_map], dim=1))

        if inj.ndim == 2:
            inj = inj.unsqueeze(0)
        if inj.size(1) < self.time_steps:
            pad = torch.zeros(b, self.time_steps - inj.size(1), inj.size(2), device=inj.device, dtype=inj.dtype)
            inj = torch.cat([inj, pad], dim=1)
        else:
            inj = inj[:, : self.time_steps]

        t_idx = torch.arange(self.time_steps, device=x.device)
        t_emb = self.t_embed(t_idx).unsqueeze(0).expand(b, self.time_steps, -1)
        depth_seq = d.unsqueeze(1).expand(-1, self.time_steps, -1)
        cond_input = torch.cat([inj, depth_seq], dim=2)
        cond_seq = t_emb + self.cond_mlp(cond_input)

        z_bt = z.unsqueeze(1).expand(b, self.time_steps, -1, h, w).reshape(b * self.time_steps, -1, h, w)
        cond_bt = cond_seq.reshape(b * self.time_steps, -1)

        for fno in self.fno_blocks:
            z_bt = fno(z_bt, cond_bt)

        return self.decoder(z_bt).view(b, self.time_steps, 2, h, w)
