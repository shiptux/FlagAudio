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
def DB_to_amplitude_triton_kernel(
    x_ptr,
    output_ptr,
    ref: float,
    power: float,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    log10 = 2.302585092994046
    x = tl.load(x_ptr + offsets, mask=mask)
    result = ref * tl.exp(0.1 * x * log10 * power)

    tl.store(output_ptr + offsets, result, mask=mask)


def DB_to_amplitude(x: torch.Tensor, ref: float, power: float) -> torch.Tensor:

    BLOCK_SIZE = 32
    n_elements = x.numel()
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)

    output = torch.empty_like(x)

    DB_to_amplitude_triton_kernel[grid_size](
        x, output, ref, power, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return output


def test_op():
    import torchaudio

    x = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float32).to("cuda")
    ref = 3.0
    power = 0.5
    output_tensor = DB_to_amplitude(x, ref, power)
    golden_tensor = torchaudio.functional.DB_to_amplitude(x, ref, power)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)
    print(f"Output tensor: {output_tensor}")
    print(f"Golden tensor: {golden_tensor}")


if __name__ == "__main__":
    test_op()
