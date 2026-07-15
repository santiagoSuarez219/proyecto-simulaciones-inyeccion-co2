import pytest
import torch
import torch.nn as nn

from fno_co2.config import Config
from fno_co2.models.registry import build_model
from fno_co2.models.variants.fno_axial_attn import AxialAttentionBlock, FNOAxialAttention
from fno_co2.training.optim import build_param_groups


class TestAxialAttentionBlock:
    """Test del bloque de atención axial (Fase 4, caso 1-2)."""

    def test_shape_preserves_square_grid(self):
        """AxialAttentionBlock preserva (N,C,H,W) en una grilla cuadrada."""
        n, c, h, w = 2, 128, 100, 100
        block = AxialAttentionBlock(c, num_heads=4, dropout_p=0.1)
        x = torch.randn(n, c, h, w)
        out = block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_shape_preserves_non_square_grid(self):
        """AxialAttentionBlock preserva (N,C,H,W) en una grilla no cuadrada."""
        n, c, h, w = 2, 128, 30, 26
        block = AxialAttentionBlock(c, num_heads=4, dropout_p=0.1)
        x = torch.randn(n, c, h, w)
        out = block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_heads_validation(self):
        """hidden_dim no divisible por attn_heads lanza error explícito."""
        c = 128
        invalid_heads = 5  # 128 % 5 != 0
        with pytest.raises(ValueError, match="hidden_dim.*divisible"):
            AxialAttentionBlock(c, num_heads=invalid_heads)

    def test_residual_connection(self):
        """AxialAttentionBlock mantiene una conexión residual."""
        n, c, h, w = 2, 128, 32, 32
        block = AxialAttentionBlock(c, num_heads=4, dropout_p=0.0)
        x = torch.randn(n, c, h, w)
        with torch.no_grad():
            out = block(x)
        # La salida debe estar "cerca" de la entrada (residual), no arbitraria
        # (sin dropout, la atención inicializada aleatoriamente no debería cambiar mucho en la primer pasada)
        diff = (out - x).norm()
        input_norm = x.norm()
        assert diff < input_norm, "Residual connection expected: output should be close to input"


class TestFNOAxialAttention:
    """Test del modelo completo FNOAxialAttention (Fase 4, caso 2-3)."""

    def test_forward_shape_square(self):
        """forward devuelve (B, T, 2, H, W) en grilla 100×100."""
        b, t, h, w = 2, 61, 100, 100
        model = FNOAxialAttention(time_steps=t, h_dim=128, attn_heads=4, attn_num_blocks=4)
        model.eval()
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)
        with torch.no_grad():
            out = model(x, d, inj)
        assert out.shape == (b, t, 2, h, w), f"Expected {(b, t, 2, h, w)}, got {out.shape}"

    def test_forward_shape_non_square(self):
        """forward devuelve (B, T, 2, H, W) en grilla no cuadrada 30×26."""
        b, t, h, w = 2, 61, 30, 26
        model = FNOAxialAttention(time_steps=t, h_dim=128, attn_heads=4, attn_num_blocks=4)
        model.eval()
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)
        with torch.no_grad():
            out = model(x, d, inj)
        assert out.shape == (b, t, 2, h, w), f"Expected {(b, t, 2, h, w)}, got {out.shape}"

    def test_backward_finite_gradients(self):
        """loss.backward() produce gradientes finitos (sin NaN/Inf)."""
        b, t, h, w = 2, 10, 32, 32
        model = FNOAxialAttention(time_steps=t, h_dim=128, attn_heads=4, attn_num_blocks=2)
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)
        y = torch.randn(b, t, 2, h, w)

        out = model(x, d, inj)
        loss = ((out - y) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"

    def test_time_steps_attribute(self):
        """El modelo expone self.time_steps."""
        t = 61
        model = FNOAxialAttention(time_steps=t)
        assert hasattr(model, "time_steps"), "Model must expose self.time_steps"
        assert model.time_steps == t

    def test_attn_num_blocks_reduced(self):
        """attn_num_blocks < 4 aplica atención solo en los últimos bloques."""
        b, t, h, w = 1, 10, 32, 32
        model = FNOAxialAttention(time_steps=t, h_dim=128, attn_heads=4, attn_num_blocks=2)
        # Verificar que hay 2 bloques de atención (None en los primeros 2)
        assert model.attn_blocks[0] is None, "First 2 blocks should be None"
        assert model.attn_blocks[1] is None, "First 2 blocks should be None"
        assert model.attn_blocks[2] is not None, "Last 2 blocks should have attention"
        assert model.attn_blocks[3] is not None, "Last 2 blocks should have attention"

        # Verificar que forward sigue funcionando
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)
        out = model(x, d, inj)
        assert out.shape == (b, t, 2, h, w)


class TestMCDropout:
    """Test de MC Dropout (Fase 4, caso 4)."""

    def test_mc_dropout_produces_uncertainty(self):
        """Con dropout_p > 0 en train mode, dos forwards difieren."""
        b, t, h, w = 2, 20, 32, 32
        model = FNOAxialAttention(
            time_steps=t, h_dim=128, attn_heads=4, attn_num_blocks=4, dropout_p=0.3
        )
        model.train()  # Forzar train mode (habilita dropout)
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)

        # Dos forwards con el mismo input
        out1 = model(x, d, inj)
        out2 = model(x, d, inj)

        # Deben diferir (sin seed fijo)
        diff = (out1 - out2).abs().sum()
        assert diff > 0, "MC Dropout: expected different outputs in two forwards with dropout_p > 0"


class TestParamGroups:
    """Test de param groups (Fase 4, caso 5)."""

    def test_param_groups_correct_classification(self):
        """build_param_groups clasifica correctamente FiLM/embeddings en no_decay."""
        model = FNOAxialAttention(time_steps=61, h_dim=128, attn_heads=4, attn_num_blocks=4)
        groups = build_param_groups(model, weight_decay=1e-4)

        assert len(groups) == 2, "Should have 2 groups: decay and no_decay"
        decay_params = set(id(p) for p in groups[0]["params"])
        no_decay_params = set(id(p) for p in groups[1]["params"])

        # Verificar que los FiLM gamma/beta están en no_decay
        for name, param in model.named_parameters():
            param_id = id(param)
            if "gamma" in name or "beta" in name:
                assert param_id in no_decay_params, f"{name} (FiLM) should be in no_decay"
            elif "t_embed" in name:
                # Embedding debe estar en no_decay
                assert param_id in no_decay_params, f"{name} (embedding) should be in no_decay"
            elif "pos_embed" in name:
                # Positional embedding debe estar en no_decay
                assert param_id in no_decay_params, f"{name} (positional embedding) should be in no_decay"
            elif ".bias" in name:
                assert param_id in no_decay_params, f"{name} (bias) should be in no_decay"

        # Verificar que hay parámetros en ambos grupos
        assert len(decay_params) > 0, "Should have decay params (Linear, Conv2d weights)"
        assert len(no_decay_params) > 0, "Should have no_decay params (bias, embeddings, FiLM)"


class TestRegistry:
    """Test del registry (Fase 4, caso 7)."""

    def test_build_model_fno_axial_attn(self):
        """build_model con model_variant='fno_axial_attn' devuelve FNOAxialAttention."""
        cfg = Config(model_variant="fno_axial_attn")
        model = build_model(cfg)
        assert isinstance(model, FNOAxialAttention), f"Expected FNOAxialAttention, got {type(model)}"

    def test_build_model_fno_baseline(self):
        """build_model con model_variant='fno_baseline' sigue devolviendo PhysicalFNOArchitecture."""
        from fno_co2.models.fno import PhysicalFNOArchitecture

        cfg = Config(model_variant="fno_baseline")
        model = build_model(cfg)
        assert isinstance(model, PhysicalFNOArchitecture), f"Expected PhysicalFNOArchitecture, got {type(model)}"

    def test_build_model_unknown_variant_raises(self):
        """build_model con variante desconocida lanza ValueError."""
        cfg = Config(model_variant="unknown_model")
        with pytest.raises(ValueError, match="desconocida"):
            build_model(cfg)


class TestConfigYAML:
    """Test de config YAML (Fase 4, sin marca explícita pero implícita en criterios)."""

    def test_yaml_round_trip(self):
        """configs/experiments/fno_axial_attn.yaml carga correctamente y produce Config válido."""
        import yaml

        yaml_path = "configs/experiments/fno_axial_attn.yaml"
        with open(yaml_path) as f:
            cfg_dict = yaml.safe_load(f)

        cfg = Config(**cfg_dict)
        assert cfg.model_variant == "fno_axial_attn"
        assert cfg.attn_heads == 4
        assert cfg.attn_num_blocks == 4

        # Verificar que build_model funciona
        model = build_model(cfg)
        assert isinstance(model, FNOAxialAttention)


class TestBaselineUnmodified:
    """Verificar que fno.py y blocks.py no fueron modificados."""

    def test_baseline_architecture_unchanged(self):
        """PhysicalFNOArchitecture funciona como antes."""
        from fno_co2.models.fno import PhysicalFNOArchitecture

        cfg = Config(model_variant="fno_baseline")
        model = PhysicalFNOArchitecture(
            time_steps=cfg.time_steps,
            in_c=5,
            h_dim=cfg.hidden_dim,
            modes=cfg.spectral_modes,
            cond_dim=128,
            dropout_p=cfg.dropout_p,
            use_group_norm=cfg.use_group_norm,
        )

        b, t, h, w = 1, 61, 32, 32
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)

        with torch.no_grad():
            out = model(x, d, inj)
        assert out.shape == (b, t, 2, h, w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
