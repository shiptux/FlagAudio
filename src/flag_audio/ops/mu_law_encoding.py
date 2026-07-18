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
def mu_law_encoding_kernel(
    x_ptr,  # 输入浮点张量指针
    out_ptr,  # 输出整数张量指针
    mu,  # μ 值 (quantization_channels - 1.0)
    n_elements,  # 总元素数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载输入（假设为 float32，也可强制转换）
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # 符号函数
    sign_x = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))

    # μ-law 编码公式
    x_mu = sign_x * tl.log(mu * tl.abs(x) + 1) / tl.log(mu + 1)

    # 量化为整数 [0, quantization_channels-1]
    x_mu = ((x_mu + 1.0) / 2.0 * mu + 0.5).to(tl.int64)

    tl.store(out_ptr + offsets, x_mu, mask=mask)


def mu_law_encoding(x: torch.Tensor, quantization_channels: int) -> torch.Tensor:
    """
    对输入张量 x (范围 [-1, 1]) 进行 μ-law 编码，返回整数张量
    """
    # 确保输入是浮点类型
    if not x.is_floating_point():
        x = x.float()
    # 将输入展平为 1D 以便于并行处理
    x_flat = x.contiguous().view(-1)
    n = x_flat.numel()

    mu = quantization_channels - 1.0
    # 分配输出张量，类型为 int64
    out = torch.empty(n, dtype=torch.int64, device=x.device)

    # 选择块大小（可调参数）
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # 启动内核
    mu_law_encoding_kernel[grid](x_flat, out, mu, n, BLOCK_SIZE=BLOCK_SIZE)

    # 恢复原始形状
    return out.view(x.shape)


def test_op():
    input_tensor = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32).to("cuda")
    quantization_channels = 256
    output_tensor = mu_law_encoding(input_tensor, quantization_channels)
    golden_tensor = torchaudio.functional.mu_law_encoding(
        input_tensor, quantization_channels
    )
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_op()
