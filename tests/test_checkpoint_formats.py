"""Tests for HybridCodec quantizer checkpoint formats."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from hybridcodec.model import SingleLayerResidualQuantizer


def _make_quantizer() -> SingleLayerResidualQuantizer:
    """Create a small quantizer for checkpoint-loading tests."""
    return SingleLayerResidualQuantizer(
        input_dim=8,
        bottleneck_dim=2,
        codebook_size=4,
        semantic_hidden_sizes=(8, 4),
        semantic_downscale_factors=(1, 1),
        residual_hidden_sizes=(4,),
        residual_downscale_factors=(1,),
    )


def test_quantizer_pretrained_hook_loads_ckpt_and_safetensors(
    tmp_path: Path,
) -> None:
    """Load equivalent parameters from both supported checkpoint formats."""
    source = _make_quantizer()
    ckpt_path = tmp_path / "quantizer.ckpt"
    safetensors_path = tmp_path / "quantizer.safetensors"
    torch.save(source.state_dict(), ckpt_path)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in source.state_dict().items()},
        str(safetensors_path),
    )

    for checkpoint_path in (ckpt_path, safetensors_path):
        loaded = _make_quantizer()
        loaded.load_pretrained(checkpoint_path)
        for key, expected in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], expected)
