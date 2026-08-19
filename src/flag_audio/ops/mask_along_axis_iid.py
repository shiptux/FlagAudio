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
from typing import Union


def _get_mask_param(mask_param: int, p: float, dim_size: int) -> int:
    """辅助函数，计算实际允许的最大掩码长度"""
    if p == 1.0:
        return mask_param
    max_v = min(mask_param, int(dim_size * p))
    return max_v


@triton.jit
def mask_along_axis_iid_kernel(
    specgrams_ptr,
    output_ptr,
    value_ptr,
    min_value_ptr,
    mask_value_ptr,
    batch_size,
    dim1,
    dim2,
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
    mask_start = tl.load(min_value_ptr + batch_id).to(tl.int64)
    mask_end = mask_start + tl.load(value_ptr + batch_id).to(tl.int64)
    mask_val = tl.load(mask_value_ptr)

    # 根据轴判断当前元素是否在掩码区间内
    if is_freq:
        in_mask = (dim1_id >= mask_start) & (dim1_id < mask_end)
    else:
        in_mask = (dim2_id >= mask_start) & (dim2_id < mask_end)

    x = tl.load(specgrams_ptr + batch_id * total_elements + offsets, mask=mask)
    y = tl.where(in_mask, mask_val.to(x.dtype), x)
    tl.store(output_ptr + batch_id * total_elements + offsets, y, mask=mask)


def mask_along_axis_iid(
    specgrams: torch.Tensor,
    mask_param: int,
    mask_value: Union[float, torch.Tensor],
    axis: int,
    p: float = 1.0,
) -> torch.Tensor:
    dim = specgrams.dim()
    if dim < 3:
        raise ValueError(
            f"Spectrogram must have at least three dimensions ({dim} given)."
        )
    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            "Only Frequency and Time masking are supported"
            f" (axis {dim - 2} and axis {dim - 1} supported; {axis} given)."
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    # 计算有效 mask_param
    axis_size = specgrams.size(axis)
    effective_mask_param = _get_mask_param(mask_param, p, axis_size)
    if effective_mask_param < 1:
        return specgrams

    # 准备批次维度：将所有前面的维度合并为批次维度
    original_shape = specgrams.shape
    specgrams_2d = specgrams.contiguous().reshape(
        -1, original_shape[-2] * original_shape[-1]
    )
    batch_size, inner_dim = specgrams_2d.shape
    # 分配输出张量
    output = torch.empty_like(specgrams_2d)

    # 生成随机值：value 和 min_value，形状 (N,)
    device = specgrams.device
    dtype = specgrams.dtype

    value = torch.rand(batch_size, device=device, dtype=dtype) * effective_mask_param
    min_value = torch.rand(batch_size, device=device, dtype=dtype) * (axis_size - value)

    if not isinstance(mask_value, torch.Tensor):
        mask_value = torch.tensor([mask_value], device=device, dtype=dtype)

    is_freq = axis == dim - 2
    BLOCK_SIZE = 16
    # 计算网格大小
    grid = (triton.cdiv(inner_dim, BLOCK_SIZE), batch_size)
    # 启动内核
    mask_along_axis_iid_kernel[grid](
        specgrams_2d,
        output,
        value,
        min_value,
        mask_value,
        batch_size,
        original_shape[-2],
        original_shape[-1],
        is_freq=is_freq,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 恢复原始形状
    return output.reshape(original_shape)


def mock_rand(size, **kwargs):
    return torch.ones(size, **kwargs) * 0.7


def test_op():
    import torchaudio

    specgram = torch.rand(2, 2, 9, 4, device="cuda")
    mask_param = 3
    mask_value = 0.0
    axis = 3
    p = 1
    torch.rand = mock_rand
    output_triton = mask_along_axis_iid(
        specgram, mask_param, mask_value, axis, p
    )
    golden_output = torchaudio.functional.mask_along_axis_iid(
        specgram, mask_param, mask_value, axis, p
    )
    torch.testing.assert_close(output_triton, golden_output, rtol=1e-5, atol=1e-8)
    print(f"Output tensor: {output_triton}")
    print(f"Golden tensor: {golden_output}")


if __name__ == "__main__":
    test_op()
