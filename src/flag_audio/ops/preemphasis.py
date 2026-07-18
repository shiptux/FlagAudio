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
def preemphasis_kernel(
    x_ptr,
    y_ptr,
    coeff: float,
    x_stride_0: tl.constexpr,
    x_stride_1: tl.constexpr,
    y_stride_0: tl.constexpr,
    y_stride_1: tl.constexpr,
    last_dim_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = tl.program_id(1)

    logic_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = logic_offsets < last_dim_len
    x = tl.load(x_ptr + row_id * x_stride_0 + logic_offsets * x_stride_1, mask=mask)

    pos = logic_offsets % last_dim_len
    prev_mask = (logic_offsets >= 1) & mask & (pos != 0)
    prev_offsets = (logic_offsets - 1) * x_stride_1

    prev_x = tl.load(
        x_ptr + row_id * x_stride_0 + prev_offsets, mask=prev_mask, other=0.0
    )
    result = tl.where(pos == 0, x, x - coeff * prev_x)
    tl.store(
        y_ptr + row_id * y_stride_0 + logic_offsets * y_stride_1, result, mask=mask
    )


def preemphasis(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    waveform_out = waveform.clone()
    waveform_view = waveform_out.reshape(-1, waveform.shape[-1])

    last_dim_len = waveform.shape[-1]
    total_elements = waveform.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(last_dim_len, BLOCK_SIZE), waveform_view.size(1))

    preemphasis_kernel[grid](
        waveform_view,
        waveform_out,
        coeff,
        waveform_view.stride(0),
        waveform_view.stride(1),
        waveform_out.stride(0),
        waveform_out.stride(1),
        last_dim_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return waveform_out


def test_op():
    waveform = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=torch.float32).to(
        "cuda"
    )
    coeff = 0.97
    output_tensor = preemphasis(waveform, coeff)
    golden_tensor = torchaudio.functional.preemphasis(waveform, coeff)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)
    print(f"Output tensor: {output_tensor}")
    print(f"Golden tensor: {golden_tensor}")


if __name__ == "__main__":
    test_op()
