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
import triton
import triton.language as tl


@triton.jit
def cal_noise_scale_kernel(
    waveform_ptr,  # (batch, L)
    noise_ptr,  # (batch, L)
    snr_ptr,  # (batch,)
    lengths_ptr,  # (batch,) or None
    scale_ptr,  # (batch,)
    L: tl.constexpr,
    has_lengths: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    batch_length = tl.load(lengths_ptr + b) if has_lengths else L
    block_cnt = (batch_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_w = 0.0
    scale_n = 0.0
    for i in range(block_cnt):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < batch_length
        w = tl.load(waveform_ptr + b * L + offset, mask=mask, other=0.0)
        scale_w += tl.sum(w * w)
        n = tl.load(noise_ptr + b * L + offset, mask=mask, other=0.0)
        scale_n += tl.sum(n * n)

    snr = (tl.log(scale_w) - tl.log(scale_n)) / tl.log(10.0) * 10
    scale = tl.exp((snr - tl.load(snr_ptr + b)) / 20 * tl.log(10.0))
    tl.store(scale_ptr + b, scale)


@triton.jit
def apply_noise_kernel(
    waveform_ptr,  # (batch, L)
    noise_ptr,  # (batch, L)
    scale_ptr,  # (batch,)
    output_ptr,  # (batch, L)
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    scale = tl.load(scale_ptr + b)
    block_start = b * L + pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (b + 1) * L
    w = tl.load(waveform_ptr + offsets, mask=mask, other=0.0)
    n = tl.load(noise_ptr + offsets, mask=mask, other=0.0)
    result = w + n * scale
    tl.store(output_ptr + offsets, result, mask=mask)


def add_noise(
    waveform: torch.Tensor,
    noise: torch.Tensor,
    snr: torch.Tensor,
    lengths: torch.Tensor = None,
) -> torch.Tensor:

    if not (
        waveform.ndim - 1 == noise.ndim - 1 == snr.ndim
        and (lengths is None or lengths.ndim == snr.ndim)
    ):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(
            f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)})."
        )
    # 展平前导维度
    total_elements = waveform.numel()
    batch_size = total_elements // L

    scale = torch.zeros(batch_size, device=waveform.device)
    BLOCK_SIZE = 256
    has_lengths = lengths is not None
    cal_noise_scale_kernel[(batch_size,)](
        waveform,
        noise,
        snr,
        lengths,
        scale,
        L,
        has_lengths,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 分配输出
    output = torch.empty_like(waveform)
    grid = (triton.cdiv(L, BLOCK_SIZE), batch_size)
    apply_noise_kernel[grid](
        waveform,
        noise,
        scale,
        output,
        L,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def test_op():
    import torchaudio

    x = torch.rand([5, 2, 2, 8], dtype=torch.float32).to("cuda")
    noise = torch.rand(x.shape, dtype=torch.float32).to("cuda")
    snr = torch.rand(x.shape[:-1], dtype=torch.float32).to("cuda")

    output_tensor = add_noise(x, noise, snr)
    golden_tensor = torchaudio.functional.add_noise(x, noise, snr)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)
    
    length = torch.ones_like(snr, dtype=torch.int64).to("cuda") * (x.shape[-1] // 2)
    output_tensor = add_noise(x, noise, snr, lengths=length)
    golden_tensor = torchaudio.functional.add_noise(x, noise, snr, lengths=length)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_op()
