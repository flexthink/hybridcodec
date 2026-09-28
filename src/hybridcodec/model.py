#!/usr/bin/env python3
"""
Hybrid Codec – Same-Layer Variant (WavLM Layer 6)

Includes code from KNNVC and FocalCodec
(TODO: References)

Authors
* Francesco Verdini 2025 (codec)
* Artem Ploujnikov 2026 (adaptation)
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from collections import namedtuple
from functools import partial
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Module
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

if TYPE_CHECKING:
    from speechbrain.integrations.huggingface.wavlm import WavLM

# =============================================================================
# Defaults – override via CLI or constructor
# =============================================================================

SAMPLE_RATE = 16000
INPUT_DIM = 1024  # WavLM L6 feature dim
CODEBOOK_SIZE = 8192  # 13-bit codes
BOTTLENECK_DIM = 32  # residual bottleneck
RESIDUAL_HIDDEN = (512, 256)
RESIDUAL_STRIDE = (1, 1)

# =============================================================================
# LFQ (Lookup-Free / Binary Spherical Quantization)
# =============================================================================

Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])


def _exists(v: object) -> bool:
    """Check whether a value is present.

    Arguments
    ---------
    v : object
        Value to check for presence.

    Returns
    -------
    bool
        Result produced by this operation.
    """
    return v is not None


def _default(*a: Any) -> Any:
    """Select the first present value, evaluating it when callable.

    Arguments
    ---------
    *a : Any
        Candidate values checked in order.

    Returns
    -------
    Any
        Result produced by this operation.
    """
    for x in a:
        if _exists(x):
            return x() if callable(x) else x
    return None


class LFQ(Module):
    def __init__(
        self,
        *,
        dim: Optional[int] = None,
        codebook_size: Optional[int] = None,
        spherical: bool = True,
        codebook_scale: float = 1.0,
        force_quantization_f32: bool = True,
        **_kw: Any,
    ):
        """Initialize the lookup-free binary quantizer.

        Arguments
        ---------
        dim : Optional[int]
            Feature dimension, inferred from the codebook size when omitted.

        codebook_size : Optional[int]
            Number of entries represented by the quantizer codebook.

        spherical : bool
            Whether to normalize vectors onto the codebook sphere.

        codebook_scale : float
            Magnitude used for quantized code vectors.

        force_quantization_f32 : bool
            Whether to run quantization calculations in float32.

        **_kw : Any
            Additional keyword arguments accepted for compatibility.
        """
        super().__init__()
        codebook_size = _default(codebook_size, lambda: 2**dim)
        codebook_dim = int(log2(codebook_size))
        dim = _default(dim, codebook_dim)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook_scale = codebook_scale
        self.spherical = spherical
        self.force_quantization_f32 = force_quantization_f32
        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        codes = torch.arange(codebook_size)
        bits = ((codes[..., None].int() & self.mask) != 0).float()
        self.register_buffer(
            "codebook", (bits * 2 - 1) * codebook_scale, persistent=False
        )

    def _l2(self, t: Tensor) -> Tensor:
        """Normalize features when spherical quantization is enabled.

        Arguments
        ---------
        t : Tensor
            Tensor to normalize when spherical quantization is enabled.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        return F.normalize(t, dim=-1) * self.codebook_scale if self.spherical else t

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Convert binary code indices to quantized vectors.

        Arguments
        ---------
        indices : Tensor
            Integer code indices to convert back to code vectors.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        bits = ((indices[..., None].int() & self.mask) != 0).to(self.codebook.dtype)
        codes = (bits * 2 - 1) * self.codebook_scale
        return self._l2(codes)

    def forward(self, x: Tensor, **_kw: Any) -> Return:
        """Quantize features and return codes with auxiliary loss.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        **_kw : Any
            Additional keyword arguments accepted for compatibility.

        Returns
        -------
        Return
            Result produced by this operation.
        """
        ctx = (
            partial(autocast, enabled=False)
            if self.force_quantization_f32
            else nullcontext
        )
        with ctx():
            if self.force_quantization_f32:
                orig = x.dtype
                x = x.float()
            x = self._l2(x)
            cv = torch.ones_like(x) * self.codebook_scale
            q = torch.where(x > 0, cv, -cv)
            indices = reduce((q > 0).int() * self.mask.int(), "b n d -> b n", "sum")
            q = self._l2(q)
            if self.training:
                x = x + (q - x).detach()
            else:
                x = q
            if self.force_quantization_f32:
                x = x.to(orig)
        return Return(x, indices, self.zero)


# =============================================================================
# FocalNet building blocks
# =============================================================================


class Snake1d(nn.Module):
    def __init__(self, dim: int):
        """Initialize the learnable Snake activation.

        Arguments
        ---------
        dim : int
            Number of input feature channels.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the learnable Snake activation to features.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha * x).sin() ** 2


class DownScale(nn.Module):
    def __init__(self, in_d: int, out_d: int, k: int = 1, s: int = 1):
        """Initialize a channel and time downscaling block.

        Arguments
        ---------
        in_d : int
            Number of input feature channels.

        out_d : int
            Number of output feature channels.

        k : int
            Convolution kernel size.

        s : int
            Convolution stride.
        """
        super().__init__()
        self.kernel_size, self.stride = k, s
        self.downscale = nn.Conv1d(in_d, out_d, k, s)
        self.activation = Snake1d(out_d)

    def forward(self, x: Tensor) -> Tensor:
        """Downscale a feature sequence and apply Snake activation.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        pad = (x.shape[-2] - self.kernel_size) % self.stride
        if pad > 0:
            x = F.pad(x, [0, 0, 0, pad])
        return self.activation(self.downscale(x.movedim(-1, -2))).movedim(-2, -1)


class UpScale(nn.Module):
    def __init__(self, in_d: int, out_d: int, k: int = 1, s: int = 1):
        """Initialize a time upscaling block.

        Arguments
        ---------
        in_d : int
            Number of input feature channels.

        out_d : int
            Number of output feature channels.

        k : int
            Convolution kernel size.

        s : int
            Convolution stride.
        """
        super().__init__()
        self.activation = Snake1d(in_d)
        self.upscale = nn.ConvTranspose1d(in_d, out_d, k, s)

    def forward(self, x: Tensor, shape: Optional[Sequence[int]] = None) -> Tensor:
        """Upscale a feature sequence to an optional target shape.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        shape : Optional[Sequence[int]]
            Optional target sequence shape used to match temporal length.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        o = self.upscale(self.activation(x.movedim(-1, -2))).movedim(-2, -1)
        if shape is not None:
            d = shape[-2] - o.shape[-2]
            if d > 0:
                o = F.pad(o, [0, 0, 0, d])
            elif d < 0:
                o = o.narrow(-2, 0, shape[-2])
        return o


class FocalModulation(nn.Module):
    def __init__(
        self,
        dim: int,
        fw: int = 7,
        fl: int = 2,
        ff: int = 2,
        drop: float = 0.0,
        post_norm: bool = False,
    ):
        """Initialize focal modulation layers.

        Arguments
        ---------
        dim : int
            Dimension of the input feature vectors.

        fw : int
            Focal convolution window width.

        fl : int
            Number of focal levels.

        ff : int
            Focal window expansion factor.

        drop : float
            Dropout probability.

        post_norm : bool
            Whether to apply normalization after each block.
        """
        super().__init__()
        self.dim, self.focal_level = dim, fl
        self.in_proj = nn.Linear(dim, 2 * dim + fl + 1)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    dim, dim, ff * k + fw, 1, groups=dim, padding=(ff * k + fw) // 2
                ),
                nn.GELU(),
            )
            for k in range(fl)
        )
        self.activation = nn.GELU()
        self.context_proj = nn.Conv1d(dim, dim, 1)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(dim) if post_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply focal context modulation to features.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        x = self.in_proj(x).movedim(-1, -2)
        q, c, g = x.split([self.dim, self.dim, self.focal_level + 1], 1)
        ca = 0.0
        for i, l in enumerate(self.layers):
            c = l(c)
            ca += c * g[:, i : i + 1]
        ca += self.activation(c.mean(-1, keepdim=True)) * g[:, self.focal_level :]
        o = (q * self.context_proj(ca)).movedim(-2, -1)
        if isinstance(self.norm, nn.LayerNorm):
            o = self.norm(o)
        return self.dropout(self.out_proj(o))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        """Initialize the two-layer feed-forward network.

        Arguments
        ---------
        dim : int
            Dimension of the input and output feature vectors.

        ffn_dim : int
            Hidden dimension of the feed-forward layer.
        """
        super().__init__()
        self.in_proj = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(ffn_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network to features.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        return self.out_proj(self.activation(self.in_proj(x)))


class FocalBlock(nn.Module):
    def __init__(self, dim: int, post_norm: bool = False):
        """Initialize a focal modulation and feed-forward block.

        Arguments
        ---------
        dim : int
            Dimension of the input feature vectors.

        post_norm : bool
            Whether to apply normalization after each block.
        """
        super().__init__()
        self.modulation_norm = nn.LayerNorm(dim)
        self.modulation = FocalModulation(dim, post_norm=post_norm)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, dim * 4)
        self.pn = post_norm

    def forward(self, x: Tensor) -> Tensor:
        """Apply the focal block with residual connections.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        h = self.modulation(x if self.pn else self.modulation_norm(x))
        if self.pn:
            h = self.modulation_norm(h)
        x = x + h
        s = x
        h = self.feed_forward(x if self.pn else self.feed_forward_norm(x))
        if self.pn:
            h = self.feed_forward_norm(h)
        return s + h


class FocalEncoder(nn.Module):
    def __init__(
        self,
        in_d: int,
        out_d: int,
        hids: Sequence[int],
        strides: Sequence[int],
        post_norm: bool = False,
    ):
        """Initialize the focal feature encoder.

        Arguments
        ---------
        in_d : int
            Number of input feature channels.

        out_d : int
            Number of output feature channels.

        hids : Sequence[int]
            Hidden channel dimensions for successive stages.

        strides : Sequence[int]
            Stride used by each successive stage.

        post_norm : bool
            Whether to apply normalization after each block.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_d
        for h, s in zip(hids, strides):
            self.layers.append(
                nn.Sequential(DownScale(d, h, s, s), FocalBlock(h, post_norm))
            )
            d = h
        self.out_proj = nn.Linear(d, out_d)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a feature sequence through focal stages.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        for l in self.layers:
            x = l(x)
        return self.out_proj(x)


class FocalDecoder(nn.Module):
    def __init__(
        self,
        in_d: int,
        out_d: int,
        hids: Sequence[int],
        strides: Sequence[int],
        post_norm: bool = False,
    ):
        """Initialize the focal feature decoder.

        Arguments
        ---------
        in_d : int
            Number of input feature channels.

        out_d : int
            Number of output feature channels.

        hids : Sequence[int]
            Hidden channel dimensions for successive stages.

        strides : Sequence[int]
            Stride used by each successive stage.

        post_norm : bool
            Whether to apply normalization after each block.
        """
        super().__init__()
        hids = tuple(hids) + (out_d,)
        self.in_proj = nn.Linear(in_d, hids[0])
        self.layers = nn.ModuleList()
        d = hids[0]
        for h, s in zip(hids[1:], strides):
            self.layers.append(
                nn.Sequential(FocalBlock(d, post_norm), UpScale(d, h, s, s))
            )
            d = h

    def forward(self, x: Tensor, shape: Optional[Sequence[int]] = None) -> Tensor:
        """Decode a feature sequence to an optional target shape.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        shape : Optional[Sequence[int]]
            Optional target sequence shape used to match temporal length.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer[1](layer[0](x), shape)
        if shape is not None:
            d = shape[-2] - x.shape[-2]
            if d > 0:
                x = F.pad(x, [0, 0, 0, d])
            elif d < 0:
                x = x.narrow(-2, 0, shape[-2])
        return x


# =============================================================================
# LookupFreeQuantizer (FocalCodec semantic branch)
# =============================================================================


class LookupFreeQuantizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: Optional[int] = 8192,
        hidden_sizes: Sequence[int] = (1024, 512, 256),
        downscale_factors: Sequence[int] = (1, 1, 1),
        use_post_norm: bool = False,
        **kw: Any,
    ):
        """Initialize the semantic lookup-free quantizer.

        Arguments
        ---------
        input_dim : int
            Dimension of the input feature vectors.

        codebook_size : Optional[int]
            Number of entries represented by the quantizer codebook.

        hidden_sizes : Sequence[int]
            Hidden dimensions used by the encoder and decoder.

        downscale_factors : Sequence[int]
            Stride factors used by the encoder and decoder.

        use_post_norm : bool
            Whether to normalize block outputs.

        **kw : Any
            Additional keyword arguments passed to the quantizer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        qdim = int(math.log2(codebook_size))
        self.quantizer = LFQ(dim=qdim, spherical=True)
        self.encoder = FocalEncoder(
            input_dim, qdim, hidden_sizes, downscale_factors, use_post_norm
        )
        self.decoder = FocalDecoder(
            qdim,
            input_dim,
            tuple(reversed(hidden_sizes)),
            tuple(reversed(downscale_factors)),
            use_post_norm,
        )

    def indices_to_codes(self, x: Tensor) -> Tensor:
        """Decode semantic indices to feature vectors.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        return self.decoder(self.quantizer.indices_to_codes(x))

    def forward(self, x: Tensor, **kw: Any) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode, quantize, and reconstruct semantic features.

        Arguments
        ---------
        x : Tensor
            Input feature tensor.

        **kw : Any
            Additional keyword arguments passed to the quantizer.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            Result produced by this operation.
        """
        z = self.encoder(x)
        q, idx, aux = self.quantizer(z, **kw)
        return idx, self.decoder(q), aux


# =============================================================================
# SingleLayerResidualQuantizer
# =============================================================================


class SingleLayerResidualQuantizer(nn.Module):
    """Discrete semantic + continuous residual on a single feature layer."""

    def __init__(
        self,
        *,
        input_dim: int = 1024,
        bottleneck_dim: int = 32,
        codebook_size: Optional[int] = 8192,
        semantic_hidden_sizes: Sequence[int] = (1024, 512, 256),
        semantic_downscale_factors: Sequence[int] = (1, 1, 1),
        residual_hidden_sizes: Sequence[int] = (512, 256),
        residual_downscale_factors: Sequence[int] = (1, 1),
        continuous_dropout: float = 0.0,
        use_post_norm: bool = False,
    ):
        """Initialize semantic and residual quantizers.

        Arguments
        ---------
        input_dim : int
            Dimension of the input feature vectors.

        bottleneck_dim : int
            Dimension of the continuous residual representation.

        codebook_size : Optional[int]
            Number of entries represented by the quantizer codebook.

        semantic_hidden_sizes : Sequence[int]
            Hidden dimensions of the semantic encoder.

        semantic_downscale_factors : Sequence[int]
            Stride factors of the semantic encoder.

        residual_hidden_sizes : Sequence[int]
            Hidden dimensions of the residual encoder.

        residual_downscale_factors : Sequence[int]
            Stride factors of the residual encoder.

        continuous_dropout : float
            Dropout probability for continuous residual features.

        use_post_norm : bool
            Whether to normalize block outputs.
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.semantic_quantizer = LookupFreeQuantizer(
            input_dim,
            codebook_size,
            semantic_hidden_sizes,
            semantic_downscale_factors,
            use_post_norm,
            spherical=True,
        )
        self.residual_encoder = FocalEncoder(
            input_dim,
            bottleneck_dim,
            residual_hidden_sizes,
            residual_downscale_factors,
            use_post_norm,
        )
        self.residual_decoder = FocalDecoder(
            bottleneck_dim,
            input_dim,
            tuple(reversed(residual_hidden_sizes)),
            tuple(reversed(residual_downscale_factors)),
            use_post_norm,
        )

        self._latents: Optional[Tensor] = None

    def forward(self, feats: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Quantize features and reconstruct semantic and residual content.

        Arguments
        ---------
        feats : Tensor
            Encoded feature sequence with shape (batch, time, feature).

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            Result produced by this operation.
        """
        idx, sem_recon, aux = self.semantic_quantizer(feats)
        T = feats.shape[1]
        if sem_recon.shape[1] != T:
            sem_recon = F.interpolate(
                sem_recon.transpose(1, 2), size=T, mode="linear", align_corners=False
            ).transpose(1, 2)
        residual = feats - sem_recon
        lat = self.residual_encoder(residual)
        self._latents = lat.detach()
        res_recon = self.residual_decoder(lat, shape=feats.shape)
        return idx, sem_recon + res_recon, aux

    def get_continuous_latents(self) -> Optional[Tensor]:
        """Return the most recently encoded continuous latents.

        Returns
        -------
        Optional[Tensor]
            Result produced by this operation.
        """
        return self._latents

    def encode_semantic(self, feats: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode features into semantic indices and reconstruction.

        Arguments
        ---------
        feats : Tensor
            Encoded feature sequence with shape (batch, time, feature).

        Returns
        -------
        Tuple[Tensor, Tensor]
            Result produced by this operation.
        """
        idx, recon, _ = self.semantic_quantizer(feats)
        return idx, recon

    def decode(
        self,
        sem_idx: Tensor,
        res_lat: Optional[Tensor],
        output_length: Optional[int] = None,
    ) -> Tensor:
        """Decode semantic indices and optional residual latents.

        Arguments
        ---------
        sem_idx : Tensor
            Semantic code indices.

        res_lat : Optional[Tensor]
            Continuous residual latents, if available.

        output_length : Optional[int]
            Requested decoded feature sequence length.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        sem = self.semantic_quantizer.indices_to_codes(sem_idx)
        shape = (
            (res_lat.shape[0], output_length, self.input_dim) if output_length else None
        )
        if res_lat is not None:
            res = self.residual_decoder(res_lat, shape=shape)
            if sem.shape[1] != res.shape[1]:
                sem = F.interpolate(
                    sem.transpose(1, 2),
                    size=res.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
        else:
            res = torch.tensor(0.0, device=sem.device)
        return sem + res


# =============================================================================
# Hybrid Codec – Same Layer
# =============================================================================


class HybridCodec(nn.Module):
    """
    Hybrid codec using a single WavLM layer for both discrete and continuous streams.

    Encoder:   WavLM Layer 6 (1024-dim, 50 Hz)
    Semantic:  LFQ quantizer  → 13-bit discrete tokens
    Residual:  FocalNet encoder/decoder → 32-dim continuous bottleneck
    Vocoder:   knn-vc HiFiGAN (original FocalCodec vocoder)
    """

    def __init__(
        self,
        encoder: nn.Module,
        quantizer: nn.Module,
        vocoder: nn.Module,
    ):
        """Initialize the hybrid codec components.

        Arguments
        ---------
        encoder : nn.Module
            Acoustic feature encoder.

        quantizer : nn.Module
            Semantic and residual quantizer.

        vocoder : nn.Module
            Waveform synthesis module.
        """
        super().__init__()
        self.vocoder = vocoder
        self.encoder = encoder
        self.quantizer = quantizer

    # ── public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self, sig: Tensor, length: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode audio into semantic indices and residual latents.

        Arguments
        ---------
        sig : Tensor
            Audio waveform batch.

        length : Optional[Tensor]
            Relative waveform lengths, if provided.

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Semantic code indices and continuous residual latents.
        """
        features = self.encoder(sig, length)

        # Quantize
        indices, reconstructed, aux_loss = self.quantizer(features)
        continuous_latents = self.quantizer.get_continuous_latents()

        return indices, continuous_latents

    @torch.no_grad()
    def decode(
        self,
        semantic_indices: Tensor,
        residual_latents: Optional[Tensor] = None,
        target_len: Optional[int] = None,
    ) -> Tensor:
        """Decode semantic indices and residual latents into audio.

        Arguments
        ---------
        semantic_indices : Tensor
            Discrete semantic code sequence.

        residual_latents : Optional[Tensor]
            Continuous residual latent sequence, if provided.

        target_len : Optional[int]
            Requested decoded feature sequence length.

        Returns
        -------
        Tensor
            Decoded audio waveform.
        """
        recon = self.quantizer.decode(
            semantic_indices, residual_latents, output_length=target_len
        )
        audio = self.vocoder(recon)
        if isinstance(audio, tuple):
            audio = audio[0]
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        return audio


class FocalCodecEncoder(nn.Module):
    """Adapt a FocalCodec encoder to the hybrid codec interface."""

    def __init__(self, model: nn.Module):
        """Initialize the wrapper around a pretrained FocalCodec encoder.

        Arguments
        ---------
        model : nn.Module
            Encoder module from a pretrained FocalCodec instance.
        """
        super().__init__()
        self.model = model

    def forward(self, sig: Tensor, length: Optional[Tensor] = None) -> Tensor:
        """Extract features from a waveform batch.

        Arguments
        ---------
        sig : Tensor
            Batch of mono waveforms.

        length : Optional[Tensor]
            Relative waveform lengths, if provided.

        Returns
        -------
        Tensor
            Encoded feature sequences.
        """
        features, *_ = self.model(sig, length)
        return features


class WavLMEncoder(nn.Module):
    def __init__(self, model: WavLM, layer_id: int):
        """Initialize a WavLM feature encoder truncated at a selected layer.

        Arguments
        ---------
        model : WavLM
            Pretrained WavLM model.

        layer_id : int
            WavLM layer whose features are extracted.
        """
        super().__init__()
        self.model = model
        self.layer_id = layer_id
        while len(self.model.encoder.layers) > layer_id:
            self.model.encoder.layers.pop(-1)
        self.eval()

    def forward(self, sig: Tensor, length: Optional[Tensor] = None) -> Tensor:
        """Extract features from a waveform batch using WavLM.

        Arguments
        ---------
        sig : Tensor
            Audio waveform batch.

        length : Optional[Tensor]
            Relative waveform lengths, if provided.

        Returns
        -------
        Tensor
            Result produced by this operation.
        """
        pm = None
        if length is not None:
            from speechbrain.dataio.dataio import length_to_mask

            pm = length_to_mask(length, sig.size(1))
        feats, _ = self.model.extract_features(sig, pm, output_layer=self.layer_id)
        return feats
