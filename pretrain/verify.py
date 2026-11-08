"""Smoke checks for the dense dual mapping and the 1M lazy path.

Run with: conda run -n fa python -m pretrain.verify
The 1M case is intentionally one sample and uses one Transformer layer to
keep verification practical; production defaults remain 12 layers.
"""

import torch

from vericomp import VeriCompConfig, VeriCompPretrainModel


def run_case(length: int, *, dense: bool, layers: int):
    cfg = VeriCompConfig(num_layers=layers, dense_weight_max_elements=300_000_000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("This verification requires a PyTorch build with Flash Attention support")
    model = VeriCompPretrainModel(cfg).to(device).eval()
    ids = torch.randint(0, 4, (1, length), device=device)
    labels = torch.full_like(ids, -100)
    labels[:, ::max(1, length // 100)] = ids[:, ::max(1, length // 100)]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                enabled=device == "cuda"):
        out = model(ids, torch.flip(ids, [-1]), labels,
                    return_dense_w=dense, return_h_up=True)
    assert out["logits"].shape == (1, length, 4)
    assert out["h_up"].shape == (1, length, cfg.d_model)
    assert out["W_norm"].shape == (1, min(length, cfg.max_tokens), length)
    expected_tokens = min(length, cfg.max_tokens)
    quota_sum = out["b_pos"].float().sum(dim=-1)
    assert torch.allclose(
        quota_sum,
        torch.full_like(quota_sum, expected_tokens),
        rtol=2e-3,
        atol=2e-2,
    ), (quota_sum, expected_tokens)
    if dense:
        expected = torch.bmm(out["W_norm"].transpose(1, 2), out["b_pos"].new_zeros((1, min(length, cfg.max_tokens), cfg.d_model)))
        assert expected.shape == out["h_up"].shape
    print(f"L={length:,}: logits={tuple(out['logits'].shape)}, W_norm={type(out['W_norm']).__name__}, params={model.num_parameters():,}")


if __name__ == "__main__":
    run_case(16_384, dense=True, layers=1)
    run_case(1_048_576, dense=False, layers=1)
    production = VeriCompPretrainModel(VeriCompConfig())
    parameter_count = production.num_parameters()
    assert 95_000_000 <= parameter_count <= 105_000_000, parameter_count
    print(f"production parameters={parameter_count:,}")
    print("dual allocation verification passed")
