"""Shape tests for the hybrid codec modules."""

from typing import Optional

import torch
from torch import Tensor, nn

from hybridcodec.model import (
    DownScale,
    HybridCodec,
    LFQ,
    SingleLayerResidualQuantizer,
    UpScale,
)


def make_quantizer() -> SingleLayerResidualQuantizer:
    """Create a small randomly initialized semantic/residual quantizer."""
    return SingleLayerResidualQuantizer(
        input_dim=8,
        bottleneck_dim=2,
        codebook_size=16,
        semantic_hidden_sizes=(8, 4),
        semantic_downscale_factors=(1, 1),
        residual_hidden_sizes=(4, 2),
        residual_downscale_factors=(1, 1),
    )


class TinyFeatureEncoder(nn.Module):
    """Project one-dimensional audio samples to a small feature space."""

    def __init__(self, feature_dim: int) -> None:
        """Initialize the input projection.

        Arguments
        ---------
        feature_dim : int
            Output feature dimension.
        """
        super().__init__()
        self.projection = nn.Linear(1, feature_dim)

    def forward(self, signal: Tensor, length: Optional[Tensor] = None) -> Tensor:
        """Project waveform samples to feature vectors.

        Arguments
        ---------
        signal : Tensor
            Batched waveform samples.

        length : Optional[Tensor]
            Optional relative waveform lengths, ignored by this test adapter.

        Returns
        -------
        Tensor
            Projected feature sequence.
        """
        del length
        return self.projection(signal.unsqueeze(-1))


class TinyVocoder(nn.Module):
    """Project feature vectors back to a single waveform channel."""

    def __init__(self, feature_dim: int) -> None:
        """Initialize the waveform projection.

        Arguments
        ---------
        feature_dim : int
            Input feature dimension.
        """
        super().__init__()
        self.projection = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Project decoded features to a one-channel waveform.

        Arguments
        ---------
        features : Tensor
            Decoded feature sequence.

        Returns
        -------
        Tensor
            Waveform batch with shape (batch, channel, time).
        """
        return self.projection(features).transpose(1, 2)


def test_lfq_quantizes_features_and_returns_expected_shapes() -> None:
    quantizer = LFQ(dim=4)
    features = torch.randn(2, 5, 4)

    result = quantizer(features)

    assert result.quantized.shape == features.shape
    assert result.indices.shape == (2, 5)
    assert result.entropy_aux_loss.shape == torch.Size([])
    assert result.indices.min() >= 0
    assert result.indices.max() < 16


def test_downscale_and_upscale_match_requested_temporal_shape() -> None:
    features = torch.randn(2, 7, 3)
    downscale = DownScale(in_d=3, out_d=4, k=2, s=2)
    upscale = UpScale(in_d=4, out_d=3, k=2, s=2)

    reduced = downscale(features)
    restored = upscale(reduced, shape=features.shape)

    assert reduced.shape == (2, 4, 4)
    assert restored.shape == features.shape


def test_residual_quantizer_forward_and_decode_shapes() -> None:
    quantizer = make_quantizer()
    features = torch.randn(2, 7, 8)

    indices, reconstruction, auxiliary_loss = quantizer(features)
    latents = quantizer.get_continuous_latents()
    decoded = quantizer.decode(indices, latents, output_length=9)

    assert indices.shape == (2, 7)
    assert reconstruction.shape == features.shape
    assert auxiliary_loss.shape == torch.Size([])
    assert latents is not None
    assert latents.shape == (2, 7, 2)
    assert decoded.shape == (2, 9, 8)


def test_hybrid_codec_encode_decode_shapes_with_fresh_modules() -> None:
    feature_dim = 8
    codec = HybridCodec(
        encoder=TinyFeatureEncoder(feature_dim),
        quantizer=make_quantizer(),
        vocoder=TinyVocoder(feature_dim),
    )
    signal = torch.randn(2, 7)

    semantic_indices, residual_latents = codec.encode(signal, length=None)
    decoded = codec.decode(semantic_indices, residual_latents, target_len=9)

    assert semantic_indices.shape == (2, 7)
    assert residual_latents is not None
    assert residual_latents.shape == (2, 7, 2)
    assert decoded.shape == (2, 9)
