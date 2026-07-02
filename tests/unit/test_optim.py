import torch

from modelo_itm.models.fno import PhysicalFNOArchitecture
from modelo_itm.training.optim import build_param_groups


def _build_model():
    return PhysicalFNOArchitecture(time_steps=4, h_dim=16, modes=4, dropout_p=0.1)


def test_build_param_groups_covers_every_trainable_parameter_exactly_once():
    model = _build_model()
    groups = build_param_groups(model, weight_decay=1e-4)

    all_ids = []
    for group in groups:
        all_ids.extend(id(p) for p in group["params"])

    expected_ids = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(all_ids) == sorted(expected_ids), "cada parametro debe aparecer exactamente una vez"


def test_build_param_groups_decay_values():
    model = _build_model()
    groups = build_param_groups(model, weight_decay=1e-4)

    decay_group = next(g for g in groups if g["weight_decay"] == 1e-4)
    no_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)

    assert len(decay_group["params"]) > 0
    assert len(no_decay_group["params"]) > 0


def test_build_param_groups_excludes_bias_embeddings_and_film_gates():
    model = _build_model()
    groups = build_param_groups(model, weight_decay=1e-4)
    no_decay_ids = {id(p) for p in groups[1]["params"]}

    for name, param in model.named_parameters():
        module_path = name.rsplit(".", 1)[0] if "." in name else ""
        should_be_no_decay = (
            name.endswith(".bias")
            or name == "t_embed.weight"
            or module_path.endswith(".gamma")
            or module_path.endswith(".beta")
        )
        if should_be_no_decay:
            assert id(param) in no_decay_ids, f"{name} deberia estar en no_decay"
        else:
            assert id(param) not in no_decay_ids, f"{name} no deberia estar en no_decay"


def test_build_param_groups_works_with_adamw():
    model = _build_model()
    groups = build_param_groups(model, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)

    x = torch.randn(2, 4, 8, 8)
    d = torch.randn(2, 1)
    inj = torch.randn(2, 4, 2)
    pred = model(x, d, inj)
    loss = pred.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # no debe lanzar errores
