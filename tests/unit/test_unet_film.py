import torch
import pytest

from fno_co2.config import Config
from fno_co2.models.registry import build_model
from fno_co2.models.variants.unet_film import UNetFiLMTemporal
from fno_co2.training.optim import build_param_groups


class TestUNetFiLMTemporal:
    """Tests unitarios para la variante UNetFiLMTemporal (spec-002 Fase 4)."""

    @pytest.fixture
    def cfg(self):
        return Config(
            time_steps=61,
            hidden_dim=128,
            dropout_p=0.1,
            use_group_norm=False,
            unet_depth=3,
            weight_decay=1e-4,
        )

    @pytest.fixture
    def model(self, cfg):
        return UNetFiLMTemporal(
            time_steps=cfg.time_steps,
            in_c=5,
            h_dim=cfg.hidden_dim,
            cond_dim=128,
            dropout_p=cfg.dropout_p,
            use_group_norm=cfg.use_group_norm,
            unet_depth=cfg.unet_depth,
        )

    def test_forward_shape_100x100(self, model):
        """1. Shape feliz: forward → (B, T, 2, H, W) con grilla 100×100."""
        b, t, h, w = 2, 61, 100, 100
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)

        out = model(x, d, inj)

        assert out.shape == (b, t, 2, h, w)
        assert out.dtype == x.dtype

    def test_forward_shape_non_power_of_2(self, model):
        """2. Reconciliación de tamaños: grilla 30×26 (no potencia-de-2)."""
        b, t, h, w = 2, 61, 30, 26
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)

        out = model(x, d, inj)

        assert out.shape == (b, t, 2, h, w)
        assert out.dtype == x.dtype

    def test_backward_finite_gradients(self, model):
        """3. Backward: loss.backward() produce gradientes finitos en todos los parámetros."""
        b, t, h, w = 2, 61, 100, 100
        x = torch.randn(b, 4, h, w, requires_grad=False)
        d = torch.randn(b, 1, requires_grad=False)
        inj = torch.randn(b, t, 2, requires_grad=False)
        y_true = torch.randn(b, t, 2, h, w)

        out = model(x, d, inj)
        loss = torch.nn.functional.mse_loss(out, y_true)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradientes no finitos en {name}"

    def test_mc_dropout_active(self, model, cfg):
        """4. MC Dropout real: con dropout_p>0 y model.train(), dos forwards difieren."""
        assert cfg.dropout_p > 0, "dropout_p debe ser > 0 para esta prueba"

        b, t, h, w = 2, 61, 100, 100
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj = torch.randn(b, t, 2)

        model.train()

        out1 = model(x, d, inj)
        out2 = model(x, d, inj)

        assert not torch.allclose(out1, out2, atol=1e-6), "MC Dropout no está activo"

    def test_param_groups_no_decay(self, model, cfg):
        """5. Param groups: gamma/beta y embeddings en no_decay."""
        param_groups = build_param_groups(model, cfg.weight_decay)

        decay_params = set()
        no_decay_params = set()

        for group in param_groups:
            for param in group["params"]:
                if group.get("weight_decay", 0) == 0:
                    no_decay_params.add(param)
                else:
                    decay_params.add(param)

        for name, param in model.named_parameters():
            if "gamma" in name or "beta" in name or "t_embed" in name:
                assert param in no_decay_params, f"{name} debería estar en no_decay"

    def test_registry_build_unet_film(self):
        """6.a Registry: build_model("unet_film") devuelve UNetFiLMTemporal."""
        cfg = Config(model_variant="unet_film")
        model = build_model(cfg)

        assert isinstance(model, UNetFiLMTemporal)

    def test_registry_unknown_variant(self):
        """6.b Registry: variante desconocida lanza ValueError."""
        cfg = Config(model_variant="unknown_variant")

        with pytest.raises(ValueError, match="model_variant desconocida"):
            build_model(cfg)

    def test_registry_fno_baseline_unchanged(self):
        """Verificación adicional: baseline intacto."""
        from fno_co2.models.fno import PhysicalFNOArchitecture

        cfg = Config(model_variant="fno_baseline")
        model = build_model(cfg)

        assert isinstance(model, PhysicalFNOArchitecture)

    def test_time_steps_attribute(self, model):
        """Verificación: self.time_steps está presente."""
        assert hasattr(model, "time_steps")
        assert model.time_steps == 61

    def test_short_injection_padding(self, model):
        """Verificación: inj más corta que time_steps se rellena."""
        b, t, h, w = 2, 100, 100, 100
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj_short = torch.randn(b, 30, 2)

        out = model(x, d, inj_short)

        assert out.shape == (b, model.time_steps, 2, h, w)

    def test_injection_truncation(self, model):
        """Verificación: inj más larga que time_steps se trunca."""
        b, t, h, w = 2, 100, 100, 100
        x = torch.randn(b, 4, h, w)
        d = torch.randn(b, 1)
        inj_long = torch.randn(b, 100, 2)

        out = model(x, d, inj_long)

        assert out.shape == (b, model.time_steps, 2, h, w)
