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
import math


def _get_mask_param(mask_param, p, dim_size):
    """计算实际可掩码的最大长度（与原始实现保持一致）"""
    if p == 1.0:
        return mask_param
    else:
        return min(mask_param, int(dim_size * p))


@triton.jit
def mask_along_axis_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    mask_start,
    mask_end,
    mask_value,
    is_freq: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total_elements = dim1 * dim2
    mask = offsets < total_elements

    dim1_id = offsets // dim2
    dim2_id = offsets % dim2

    # 根据轴判断当前元素是否在掩码区间内
    if is_freq:
        in_mask = (dim1_id >= mask_start) & (dim1_id < mask_end)
    else:
        in_mask = (dim2_id >= mask_start) & (dim2_id < mask_end)

    x = tl.load(input_ptr + batch_id * total_elements + offsets, mask=mask)
    y = tl.where(in_mask, mask_value.to(x.dtype), x)
    tl.store(output_ptr + batch_id * total_elements + offsets, y, mask=mask)


def mask_along_axis(
    specgram: torch.Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> torch.Tensor:
    dim = specgram.dim()
    if dim < 2:
        raise ValueError(
            f"Spectrogram must have at least two dimensions ({dim} given)."
        )
    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            "Only Frequency and Time masking are supported "
            f"(axis {dim-2} and {dim-1} supported; {axis} given)."
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be between 0.0 and 1.0 ({p} given).")

    # 计算实际可掩码的最大长度
    max_v = _get_mask_param(mask_param, p, specgram.shape[axis])
    if max_v < 1:
        return specgram

    # 随机采样掩码长度 v 和起始位置 v0
    device = specgram.device
    v = torch.rand(1, device=device).item() * max_v
    v0 = torch.rand(1, device=device).item() * (specgram.shape[axis] - v)
    mask_start = int(v0)
    mask_end = int(v0) + int(v)

    # 重塑为 2D: (batch, dim1 * dim2)
    original_shape = specgram.shape
    specgram_2d = specgram.contiguous().reshape(
        -1, original_shape[-2] * original_shape[-1]
    )
    batch_size, inner_dim = specgram_2d.shape

    # 确定掩码轴在 2D 张量中的对应维度
    is_freq = axis == dim - 2

    # 分配输出张量
    output = torch.empty_like(specgram_2d)

    # 配置内核启动
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(inner_dim, BLOCK_SIZE), batch_size)

    # 调用内核
    mask_along_axis_kernel[grid](
        specgram_2d,
        output,
        batch_size,
        original_shape[-2],
        original_shape[-1],
        mask_start,
        mask_end,
        mask_value,
        is_freq,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 恢复原始形状
    return output.reshape(original_shape)


def mock_rand(size, **kwargs):
    return torch.ones(size, **kwargs) * 0.6


def test_op():
    import torchaudio

    specgram = torch.rand(2, 2, 4, 4, device="cuda")
    mask_param = 6
    mask_value = 0.0
    axis = 2
    p = 0.5
    torch.rand = mock_rand
    output_triton = mask_along_axis(specgram, mask_param, mask_value, axis, p)
    glolden_output = torchaudio.functional.mask_along_axis(
        specgram, mask_param, mask_value, axis, p
    )
    torch.testing.assert_close(output_triton, glolden_output, rtol=1e-5, atol=1e-8)
    print(f"Output tensor: {output_triton}")
    print(f"Golden tensor: {glolden_output}")


if __name__ == "__main__":
    test_op()
