import torch
import torch.nn as nn
import torch.nn.functional as F

from fno_co2.models.blocks import FiLMSpectralBlock, ResBlock


class AxialAttentionBlock(nn.Module):
    """Bloque de atención axial (Fase 1 del spec-003): factoriza la atención en filas y columnas.
    Entrada/salida: (N, C, H, W) sin cambios de shape.

    Componentes:
    - GroupNorm por eje (estable sin depender del batch size B*T)
    - Positional encoding aprendido por eje (H y W como nn.Embedding)
    - Atención multi-cabeza por filas (eje H) + columnas (eje W)
    - Residual + dropout
    """

    def __init__(self, c: int, num_heads: int = 4, dropout_p: float = 0.1, use_group_norm: bool = False):
        super().__init__()
        if c % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({c}) must be divisible by attn_heads ({num_heads}). "
                f"Chose attn_heads that divides hidden_dim evenly."
            )
        self.c = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads

        # GroupNorm antes de la atención
        num_groups = self._group_norm_num_groups(c)
        self.norm = nn.GroupNorm(num_groups, c)

        # Positional encoding aprendido por eje (máximo 256 en cada dimensión)
        self.pos_embed_h = nn.Embedding(256, c)
        self.pos_embed_w = nn.Embedding(256, c)

        # Atención por filas
        self.attn_h = nn.MultiheadAttention(c, num_heads, dropout=dropout_p, batch_first=True)
        # Atención por columnas
        self.attn_w = nn.MultiheadAttention(c, num_heads, dropout=dropout_p, batch_first=True)

        self.dropout = nn.Dropout(dropout_p)

    @staticmethod
    def _group_norm_num_groups(c: int, preferred: int = 8) -> int:
        for g in (preferred, 4, 2, 1):
            if c % g == 0:
                return g
        return 1

    def forward(self, x):
        """x: (N, C, H, W) → (N, C, H, W)"""
        n, c, h, w = x.shape
        residual = x

        # Normalización
        x = self.norm(x)  # (N, C, H, W)

        # Positional encoding por eje
        h_pos = self.pos_embed_h(torch.arange(h, device=x.device))  # (H, C)
        w_pos = self.pos_embed_w(torch.arange(w, device=x.device))  # (W, C)

        # ============ Atención por filas (eje H) ============
        # Reordenar a (N*W, H, C) para tratar cada columna como una secuencia de H tokens
        x_h = x.permute(0, 3, 2, 1)  # (N, W, H, C)
        x_h = x_h.reshape(n * w, h, c)  # (N*W, H, C)

        # Sumar positional encoding para la dimensión H
        x_h = x_h + h_pos.unsqueeze(0)  # (N*W, H, C)

        # Multi-head self-attention
        attn_out_h, _ = self.attn_h(x_h, x_h, x_h)  # (N*W, H, C)
        attn_out_h = attn_out_h.reshape(n, w, h, c).permute(0, 3, 2, 1)  # (N, C, H, W)

        # ============ Atención por columnas (eje W) ============
        # Reordenar a (N*H, W, C) para tratar cada fila como una secuencia de W tokens
        x_w = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x_w = x_w.reshape(n * h, w, c)  # (N*H, W, C)

        # Sumar positional encoding para la dimensión W
        x_w = x_w + w_pos.unsqueeze(0)  # (N*H, W, C)

        # Multi-head self-attention
        attn_out_w, _ = self.attn_w(x_w, x_w, x_w)  # (N*H, W, C)
        attn_out_w = attn_out_w.reshape(n, h, w, c).permute(0, 3, 1, 2)  # (N, C, H, W)

        # Promedio de ambas atenciones + residual + dropout
        attn_out = (attn_out_h + attn_out_w) / 2.0
        out = residual + self.dropout(attn_out)
        return out


class FNOAxialAttention(nn.Module):
    """Variante FNO con atención espacial axial intercalada (spec-003 Fase 2).

    Idéntico al baseline salvo que intercala AxialAttentionBlock tras cada FiLMSpectralBlock.
    El condicionamiento temporal (FiLM) sigue entrando en los bloques espectrales, sin cambios.

    Hiperparámetros:
    - attn_heads: número de cabezas en la atención (default 4)
    - attn_num_blocks: cuántos de los 4 bloques llevan atención intercalada (default 4)
    """

    def __init__(
        self,
        time_steps: int = 61,
        in_c: int = 5,
        h_dim: int = 128,
        modes: int = 16,
        cond_dim: int = 128,
        dropout_p: float = 0.1,
        use_group_norm: bool = False,
        attn_heads: int = 4,
        attn_num_blocks: int = 4,
    ):
        super().__init__()
        self.time_steps = int(time_steps)
        self.attn_num_blocks = min(attn_num_blocks, 4)  # Máximo 4 bloques

        # Encoder (idéntico al baseline)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, h_dim, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            ResBlock(h_dim, dropout_p=dropout_p, use_group_norm=use_group_norm),
        )

        # Condicionamiento temporal (idéntico al baseline)
        self.t_embed = nn.Embedding(self.time_steps, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FNO blocks (4 bloques espectrales)
        self.fno_blocks = nn.ModuleList([FiLMSpectralBlock(h_dim, modes, cond_dim=cond_dim) for _ in range(4)])

        # Bloques de atención axial (intercalados en los últimos attn_num_blocks)
        self.attn_blocks = nn.ModuleList(
            [AxialAttentionBlock(h_dim, num_heads=attn_heads, dropout_p=dropout_p, use_group_norm=use_group_norm)
             if i >= (4 - self.attn_num_blocks) else None for i in range(4)]
        )

        # Decoder (idéntico al baseline)
        self.decoder = nn.Sequential(
            ResBlock(h_dim, dropout_p=dropout_p, use_group_norm=use_group_norm),
            nn.Conv2d(h_dim, h_dim // 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(h_dim // 2, 2, 1),
        )

    def forward(self, x, d, inj):
        """Entrada idéntica al baseline: x=(B,4,H,W), d=(B,1), inj=(B,T,2).
        Salida: (B, T, 2, H, W).
        """
        b, _, h, w = x.shape

        # Encoder con depth_map (idéntico al baseline)
        depth_map = d.view(b, 1, 1, 1).expand(b, 1, h, w)
        z = self.encoder(torch.cat([x, depth_map], dim=1))

        # Condicionamiento temporal (idéntico al baseline)
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

        # Expandir z a (B*T, C, H, W)
        z_bt = z.unsqueeze(1).expand(b, self.time_steps, -1, h, w).reshape(b * self.time_steps, -1, h, w)
        cond_bt = cond_seq.reshape(b * self.time_steps, -1)

        # Núcleo: intercalar FNO blocks y atención blocks
        for i, (fno, attn) in enumerate(zip(self.fno_blocks, self.attn_blocks)):
            z_bt = fno(z_bt, cond_bt)
            if attn is not None:
                z_bt = attn(z_bt)

        # Decoder
        return self.decoder(z_bt).view(b, self.time_steps, 2, h, w)


def build(cfg) -> nn.Module:
    """Función de registro automático (spec-003 Fase 3).
    Construye FNOAxialAttention leyendo campos de Config."""
    return FNOAxialAttention(
        time_steps=cfg.time_steps,
        in_c=5,
        h_dim=cfg.hidden_dim,
        modes=cfg.spectral_modes,
        cond_dim=128,
        dropout_p=cfg.dropout_p,
        use_group_norm=cfg.use_group_norm,
        attn_heads=cfg.attn_heads,
        attn_num_blocks=cfg.attn_num_blocks,
    )
