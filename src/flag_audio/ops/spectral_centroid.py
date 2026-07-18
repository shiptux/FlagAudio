# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchaudio
import triton
import triton.language as tl


@triton.jit
def spectral_centroid_kernel(
    spec_r_ptr,  # (batch, L)
    spec_i_ptr,  # (batch, L)
    frqs_ptr,  # (fft_bins,)
    output_ptr,  # (batch,)
    fft_bins: tl.constexpr,  # int
    spec_r_stride: tl.constexpr,  # int
    spec_i_stride: tl.constexpr,  # int
    BLOCK_SIZE: tl.constexpr,  # int
):
    b = tl.program_id(0)
    block_cnt = (fft_bins + BLOCK_SIZE - 1) // BLOCK_SIZE
    div_0 = 0.0
    div_1 = 0.0
    for i in range(block_cnt):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < fft_bins
        spec_r = tl.load(
            spec_r_ptr + (b * fft_bins + offset) * spec_r_stride, mask=mask, other=0.0
        )
        spec_i = tl.load(
            spec_i_ptr + (b * fft_bins + offset) * spec_i_stride, mask=mask, other=0.0
        )
        spec = tl.sqrt(spec_r * spec_r + spec_i * spec_i)
        freqs = tl.load(frqs_ptr + offset, mask=mask, other=0.0)
        div_0 += tl.sum(freqs * spec)
        div_1 += tl.sum(spec)
    centroid = div_0 / div_1 if div_1 != 0.0 else 0.0
    tl.store(output_ptr + b, centroid)


def spectral_centroid(
    waveform: torch.Tensor,
    sample_rate: int,
    pad: int,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
):
    if pad > 0:
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    out = torch.empty(
        shape[:-1] + spec_f.shape[-1:], dtype=waveform.dtype, device=waveform.device
    )
    spec_f = spec_f.transpose(-2, -1).contiguous()
    spec_r = spec_f.real
    spec_i = spec_f.imag

    fft_bins = 1 + n_fft // 2
    freqs = torch.linspace(
        0, sample_rate // 2, steps=fft_bins, device=spec_f.device
    ).reshape((-1, 1))
    BLOCK_SIZE = 256
    grid = (spec_f.numel() // fft_bins,)
    spectral_centroid_kernel[grid](
        spec_r,
        spec_i,
        freqs,
        out,
        fft_bins,
        spec_r.stride(-1),
        spec_i.stride(-1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def test_op():
    # 1D 波形 (time,)
    waveform_1d = torch.randn(1000).to("cuda")
    sample_rate = 16000
    pad = 1
    window = torch.hann_window(400).to("cuda")
    n_fft = 512
    hop_length = 160
    win_length = 400

    output_tensor = spectral_centroid(
        waveform_1d, sample_rate, pad, window, n_fft, hop_length, win_length
    )
    golden_tensor = torchaudio.functional.spectral_centroid(
        waveform_1d, sample_rate, pad, window, n_fft, hop_length, win_length
    )
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)

    # 2D 波形 (channels, time)
    waveform_2d = torch.randn(2, 1000).to("cuda")

    output_tensor = spectral_centroid(
        waveform_2d, sample_rate, pad, window, n_fft, hop_length, win_length
    )
    golden_tensor = torchaudio.functional.spectral_centroid(
        waveform_2d, sample_rate, pad, window, n_fft, hop_length, win_length
    )
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)

    # 更高维度 (batch, channels, time)
    waveform_3d = torch.randn(3, 2, 1000).to("cuda")

    output_tensor = spectral_centroid(
        waveform_3d, sample_rate, pad, window, n_fft, hop_length, win_length
    )
    golden_tensor = torchaudio.functional.spectral_centroid(
        waveform_3d, sample_rate, pad, window, n_fft, hop_length, win_length
    )
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_op()
