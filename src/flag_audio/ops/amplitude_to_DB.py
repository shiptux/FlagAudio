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
def _amplitude_to_db_kernel(
    x_ptr,
    out_ptr,
    batch_len,
    multiplier,
    amin,
    db_multiplier,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel without top_db clipping.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_len

    x = tl.load(x_ptr + b * batch_len + offsets, mask=mask)
    x_clamped = tl.maximum(x, amin)
    log10_x = tl.log(x_clamped) / tl.log(10.0)
    x_db = multiplier * log10_x - multiplier * db_multiplier

    tl.store(out_ptr + b * batch_len + offsets, x_db, mask=mask)


@triton.jit
def _amplitude_to_db_topdb_kernel(
    x_ptr,
    batch_len,
    top_db,
    multiplier,
    amin,
    db_multiplier,
    max_vals_ptr,  # shape (B,)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel with top_db clipping. Thresholds are per batch element.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_len

    x = tl.load(x_ptr + b * batch_len + offsets, mask=mask)
    thresh = tl.load(max_vals_ptr + b) - top_db
    x_db = tl.maximum(x, thresh)

    tl.store(x_ptr + b * batch_len + offsets, x_db, mask=mask)


def amplitude_to_DB(
    x: torch.Tensor,
    multiplier: float,
    amin: float,
    db_multiplier: float,
    top_db: float = None,
) -> torch.Tensor:
    """
    Triton implementation of torchaudio.functional.amplitude_to_DB.
    Supports arbitrary input shapes: (freq, time), (channel, freq, time),
    or (..., batch, channel, freq, time). For higher dimensions, all leading
    dimensions are collapsed into a single batch dimension.
    """
    original_shape = x.shape
    ndim = x.dim()
    total = x.numel()
    packed_channels = original_shape[-3] if ndim > 2 else 1
    x_4d = x.reshape(-1, packed_channels, original_shape[-2], original_shape[-1])
    B = x_4d.shape[0]
    batch_len = total // B

    out = torch.empty_like(x)
    block_size = 512

    grid = (triton.cdiv(batch_len, block_size), B)
    _amplitude_to_db_kernel[grid](
        x_4d,
        out,
        batch_len,
        multiplier,
        amin,
        db_multiplier,
        BLOCK_SIZE=block_size,
    )
    if top_db is None:
        return out.view(original_shape)

    # Compute per‑batch thresholds for clipping
    max_vals = out.amax(dim=(-3, -2, -1))  # shape (B,)
    _amplitude_to_db_topdb_kernel[grid](
        out,
        batch_len,
        top_db,
        multiplier,
        amin,
        db_multiplier,
        max_vals,
        BLOCK_SIZE=block_size,
    )

    # Restore original shape
    out = out.view(original_shape)
    return out


def test_op():
    x = torch.empty([5, 2, 2, 2], dtype=torch.float32).to("cuda")
    multiplier = 3.0
    amin = 1e-5
    db_multiplier = 20.0
    top_db = 0.1
    output_tensor = amplitude_to_DB(x, multiplier, amin, db_multiplier)
    golden_tensor = torchaudio.functional.amplitude_to_DB(
        x, multiplier, amin, db_multiplier
    )
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)
    output_tensor = amplitude_to_DB(x, multiplier, amin, db_multiplier, top_db=top_db)
    golden_tensor = torchaudio.functional.amplitude_to_DB(
        x, multiplier, amin, db_multiplier, top_db=top_db
    )
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_op()
