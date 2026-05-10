# Data Engineering & Distillation Pipeline

This directory contains the logic for high-volume data acquisition, cleaning, and automated labeling for the CUDA-to-HIP transition.

## Overview
The goal of this module is to transform raw, heterogeneous CUDA source code into a structured dataset compatible with Fine-Tuning. We focus on "Distillation," using a larger model (Qwen2.5-Coder-7B) to act as a teacher for architectural analysis and reasoning.

## Pipeline Stages
1. **Massive Sourcing:** Automated cloning of 20+ industry-leading NVIDIA repositories (`CUTLASS`, `cuFFT`, `Megatron-LM`, etc.).
2. **Regex Extraction:** Deep parsing of `.cu` files to isolate functional `__global__` and `__device__` kernels.
3. **Automated Conversion:** Direct syntax mapping using the `HIPIFY-perl` toolchain.
4. **CoT & RAFT Labeling:** 
   - **CoT (Chain-of-Thought):** Generates step-by-step migration notes.
   - **RAFT (Reasoning-Aware):** Injects documentation context to train the model on retrieval-augmented generation patterns.

## Dataset Outputs
- `cuda_to_hip_dataset.jsonl`: Raw translation pairs.
- `cuda_hip_raft_dataset.jsonl`: Reasoning-aware samples with documentation context.
- `cuda_arch_dataset.jsonl`: Deep architectural profiles of high-performance kernels.

## Hardware for Generation
- **Environment:** 2x NVIDIA T4 GPUs.
- **Optimization:** Utilizes `device_map="auto"` and batch inference to process >1,000 kernels in under 2 hours.