"""SpeechBrain inference interface for pretrained hybrid codecs.

Authors
* Francesco Verdini 2025 (codec)
* Artem Ploujnikov 2026 (adaptation)
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from speechbrain.inference.interfaces import Pretrained


class HybridCodec(Pretrained):
    """Run a pretrained hybrid codec on batched audio or codec tokens.

    The HyperPyYAML file used with :meth:`from_hparams` must define a
    ``codec`` entry in ``modules``. Audio inputs should be mono waveforms at
    the sample rate expected by that codec.

    Example
    -------
    >>> codec = HybridCodec.from_hparams(source="path/to/codec")
    >>> waveform = codec.load_audio("audio.wav").squeeze(-1)
    >>> signal = waveform.unsqueeze(0)
    >>> lengths = torch.ones(signal.shape[0])
    >>> semantic, residual = codec.encode(signal, lengths)
    >>> reconstruction = codec.decode(semantic, residual)
    """

    MODULES_NEEDED = ["codec"]

    @torch.no_grad()
    def encode(
        self, sig: Tensor, length: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode a batch of waveforms into semantic and residual streams.

        Arguments
        ---------
        sig : Tensor
            Batch of mono waveforms with shape (batch, time).

        length : Optional[Tensor]
            Relative waveform lengths with shape (batch,), if the batch is
            padded.

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Semantic code indices and continuous residual latents.
        """
        sig = sig.to(self.device)
        if length is not None:
            length = length.to(self.device)
        with self.inference_ctx:
            return self.mods.codec.encode(sig=sig, length=length)

    @torch.no_grad()
    def decode(
        self,
        semantic_indices: Tensor,
        residual_latents: Optional[Tensor] = None,
        target_len: Optional[int] = None,
    ) -> Tensor:
        """Synthesize a waveform from semantic and residual codec streams.

        Arguments
        ---------
        semantic_indices : Tensor
            Batch of discrete semantic code sequences.

        residual_latents : Optional[Tensor]
            Continuous residual latent sequences, if available.

        target_len : Optional[int]
            Requested decoded feature sequence length.

        Returns
        -------
        Tensor
            Reconstructed waveforms with shape (batch, time).
        """
        semantic_indices = semantic_indices.to(self.device)
        if residual_latents is not None:
            residual_latents = residual_latents.to(self.device)
        with self.inference_ctx:
            return self.mods.codec.decode(
                semantic_indices=semantic_indices,
                residual_latents=residual_latents,
                target_len=target_len,
            )

    def forward(
        self, sig: Tensor, length: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode waveforms, following the standard Pretrained interface.

        Arguments
        ---------
        sig : Tensor
            Batch of mono waveforms with shape (batch, time).

        length : Optional[Tensor]
            Relative waveform lengths with shape (batch,), if the batch is
            padded.

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Semantic code indices and continuous residual latents.
        """
        return self.encode(sig, length)
