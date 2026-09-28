# HybridCodec

HybridCodec encodes speech into a discrete semantic stream and an optional
continuous residual stream, then decodes those streams back into audio.

## Load a pretrained codec

Install this package and its dependencies, then load the model from its local
Hugging Face package directory or Hub repository:

```python
import torch
from hybridcodec.inference import HybridCodec

codec = HybridCodec.from_hparams(source="hybridcodec/hybridcodec_finetune_12_5hz")
# For a local package directory, use:
# codec = HybridCodec.from_hparams(source="/path/to/hybridcodec")

waveform = codec.load_audio("sample.wav").squeeze(-1)
signal = waveform.unsqueeze(0)  # batch, time
lengths = torch.ones(signal.shape[0])
semantic, residual = codec.encode(signal, lengths)
reconstruction = codec.decode(semantic, residual)
```

`load_audio` converts input audio to mono at 16 kHz, the codec's expected sample
rate. For batched padded input, pass relative waveform lengths to `encode`.
`decode` returns waveforms with shape `(batch, time)`; `target_len` can be
provided when a particular decoded feature length is needed.

The Hugging Face model package contains HyperPyYAML configuration and
quantizer checkpoints. The configuration loads the pretrained FocalCodec
encoder and decoder from Torch Hub on first use.

## Paper

Please cite our Interspeech 2026 paper:

[HybridCodec: Modeling Discrete and Continuous Representations for Efficient Speech Language Models](https://www.isca-archive.org/interspeech_2026/ploujnikov26_interspeech.html#)

```bibtex
@inproceedings{ploujnikov26_interspeech,
  title     = {{HybridCodec: Modeling Discrete and Continuous Representations For Efficient Speech Language Models}},
  author    = {Artem Ploujnikov and Francesco Verdini and Samir Sadok and Mirco Ravanelli},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {2730--2735},
  doi       = {10.21437/Interspeech.2026-2784},
  issn      = {2958-1796},
}
```

This implementation makes use of the SpeechBrain framework. If you use HybridCodec in your research or applications, please also cite SpeechBrain:

```bibtex
@article{JMLR:v25:24-0991,
  author  = {Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Ha Nguyen and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Ga{{\"e}}lle Laperri{{\`e}}re and Mickael Rouvier and Renato De Mori and Yannick Est{{\`e}}ve},
  title   = {Open-Source Conversational AI with SpeechBrain 1.0},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {333},
  pages   = {1--11},
  url     = {http://jmlr.org/papers/v25/24-0991.html}
}
```