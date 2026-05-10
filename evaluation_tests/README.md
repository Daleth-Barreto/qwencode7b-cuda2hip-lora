# Test Suite: CUDA-to-HIP Translation

This directory contains the evaluation notebooks used to validate the **Qwen-CUDA2HIP-LoRA** adapter. The tests range from basic syntax mapping to advanced hardware-specific optimizations.

## Environment Specifications
* **Architecture:** AMD CDNA 3 (Instinct MI300X)
* **Software Stack:** ROCm v7.2.3
* **Base LLM:** Qwen2.5-Coder-7B-Instruct
* **PEFT Method:** LoRA (Rank 16, Alpha 32)

## File Manifest

| File | Description |
| :--- | :--- |
| `first_lora_test.ipynb` | Initial validation of PEFT loading and basic CUDA runtime API mapping. |
| `FunctionalLoraTest.ipynb` | Advanced functional testing, including kernel vectorization and MI300X optimization logic. |

## Testing Methodology
The adapters are tested against three main criteria:
1. **API Coverage:** Correctness in mapping `cuda*` calls to `hip*`.
2. **Kernel Semantics:** Maintaining thread/block logic during architectural porting.
3. **Hardware Optimization:** Leveraging AMD-specific types (e.g., `float4`) for memory throughput efficiency.

## Key Findings
- **Zero-Shot Portability:** The adapter significantly reduces the "hallucination" of NVIDIA-only libraries.
- **Hardware Awareness:** Successful implementation of vectorization patterns for MI300X.
- **Strict Output:** High adherence to system prompts requiring raw code output for CI/CD integration.