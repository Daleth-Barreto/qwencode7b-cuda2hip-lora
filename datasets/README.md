# Core GPU Datasets

This folder contains the results of our large-scale data engineering pipeline. These datasets provide the foundational "knowledge" required for the CUDA2HIP LoRA adapter.

## Dataset: `cuda-hip-gpu-dataset`

This is our primary supervised fine-tuning (SFT) source. It contains real-world code snippets translated and explained by experts (LLM-Distilled).

### Key Features
* **Samples:** 1,133 verified kernel pairs.
* **Source Diversity:** Includes kernels from `cuFFT`, `cuPQC`, `MathDx`, and `nvCOMP`.
* **Reasoning Depth:** Every sample includes a `Migration Notes` section explaining API shifts like `cudaMalloc` to `hipMalloc` and kernel launch macro differences.

### Data Fields
- `instruction`: The task description.
- `input`: Original NVIDIA CUDA code.
- `output`: HIP code + CoT (Chain of Thought) reasoning.
- `source`: Origin repository for traceability.

## Processing Pipeline
1. **Scraping:** Automated cloning of NVIDIA/HPC repositories.
2. **Cleaning:** Removing host-only code and non-functional boilerplate.
3. **HIPIFY:** Generating the raw HIP translation.
4. **CoT Labeling:** Generating reasoning traces using Qwen2.5-Coder on 2xT4 GPUs.

## Usage
This dataset is designed to be used with the `SFTTrainer` from the `trl` library to teach LLMs the structural and semantic rules of ROCm portability.