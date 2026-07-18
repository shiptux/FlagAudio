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
def dcshift_kernel(
    input_ptr,
    output_ptr,
    shift: tl.constexpr,
    limiter_gain: tl.constexpr,
    use_limiter: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    if use_limiter:
        abs_shift = tl.abs(shift)
        limiter_threshold = 1.0 - (abs_shift - limiter_gain)

        if shift > 0:
            # 正偏移：压缩超过阈值的部分
            diff = x - limiter_threshold
            temp = diff * limiter_gain / (1.0 - limiter_threshold)
            y1 = tl.minimum(
                temp + limiter_threshold + shift, limiter_threshold
            )  # clamp max
            y2 = tl.minimum(tl.maximum(x + shift, -1.0), 1.0)  # clamp [-1, 1]
            cond = x > limiter_threshold
            y = tl.where(cond, y1, y2)

        elif shift < 0:
            # 负偏移：压缩低于负阈值的部分
            diff_neg = x + limiter_threshold
            temp = diff_neg * limiter_gain / (1.0 - limiter_threshold)
            y1 = tl.maximum(
                temp - limiter_threshold + shift, -limiter_threshold
            )  # clamp min
            y2 = tl.minimum(tl.maximum(x + shift, -1.0), 1.0)  # clamp [-1, 1]
            cond = x < -limiter_threshold
            y = tl.where(cond, y1, y2)

        else:  # shift == 0
            y = tl.minimum(tl.maximum(x, -1.0), 1.0)

    else:
        y = x + shift
        y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    tl.store(output_ptr + offsets, y, mask=mask)


def dcshift(
    waveform: torch.Tensor,
    shift: float,
    limiter_gain: float | None = None,
) -> torch.Tensor:
    output = torch.empty_like(waveform)
    n_elements = waveform.numel()

    use_limiter = 1 if limiter_gain is not None else 0
    # 当 limiter_gain 为 None 时，传一个占位值（不会被使用）
    gain = limiter_gain if limiter_gain is not None else 0.0

    # 配置网格和块大小
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    dcshift_kernel[grid](
        waveform,
        output,
        shift,
        gain,
        use_limiter,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 保持原始形状
    return output.view_as(waveform)


def test_op():
    input_tensor = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32).to("cuda")
    shift = 0.5
    limiter_gain = 0.8
    output_tensor = dcshift(input_tensor, shift)
    golden_tensor = torchaudio.functional.dcshift(input_tensor, shift)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)
    output_tensor = dcshift(input_tensor, shift, limiter_gain=limiter_gain)
    golden_tensor = torchaudio.functional.dcshift(
        input_tensor, shift, limiter_gain=limiter_gain
    )
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_op()
