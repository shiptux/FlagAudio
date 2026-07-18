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
def gain_kernel(
    input_ptr,
    output_ptr,
    gain_db: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    ratio = 10 ** (gain_db / 20)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = input_vals * ratio

    tl.store(output_ptr + offsets, output_vals, mask=mask)


def gain(input_tensor: torch.Tensor, gain_db: float = 1.0) -> torch.Tensor:
    if gain_db == 1.0:
        return input_tensor

    output_tensor = torch.empty_like(input_tensor)

    BLOCK_SIZE = 32
    n_elements = input_tensor.numel()
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gain_kernel[grid_size](
        input_tensor,
        output_tensor,
        gain_db=gain_db,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor


def test_op():
    input_tensor = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32).to("cuda")
    gain_db = 6.0
    output_tensor = gain(input_tensor, gain_db)
    golden_tensor = torchaudio.functional.gain(input_tensor, gain_db)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)
    print(f"Output tensor: {output_tensor}")
    print(f"Golden tensor: {golden_tensor}")


if __name__ == "__main__":
    test_op()
