# CUDA2HIP-LoRA: Intelligent Portability Adapter for AMD ROCm

**An AI-driven assistant designed to bridge the gap between NVIDIA's proprietary ecosystem and AMD's open ROCm platform.**

## Project Vision
Migrating legacy CUDA codebases to ROCm is a significant barrier for HPC developers. This project introduces a **LoRA (Low-Rank Adaptation)** adapter for the `Qwen2.5-Coder-7B` model, specifically trained to not only translate code but also provide **Architectural Reasoning** and **Chain-of-Thought** migration paths for the AMD MI300X hardware.

## System Architecture
The project is divided into three core pillars:

1. **`data_engineering/`**: A massive ingestion pipeline that distilled knowledge from 5,000+ NVIDIA source files into high-quality reasoning datasets.
2. **`lora_training/`**: Specialized training module utilizing **RAFT** (Reasoning-Aware Fine-Tuning) on AMD MI300X to ensure hardware-aware optimizations (like `float4` vectorization).
3. **`test/`**: A rigorous evaluation suite verifying the adapter's performance on complex kernels from `cuBLASDx`, `cuPQC`, and `Flash-Attention`.

## Tech Stack
- **Base Model:** Qwen2.5-Coder-7B-Instruct.
- **Hardware:** AMD Instinct MI300X (CDNA 3).
- **Software:** ROCm v7.2.3, HIP, PyTorch, PEFT/LoRA.
- **Tools:** HIPIFY, BitsAndBytes (ROCm version), TRL.

## Repository Structure
```text
.
├── datasets/             # Massive data ingestion & RAFT labeling
├── lora_training/        # Fine-tuning logic on MI300X (RAFT & SFT)
├── test/                 # Functional & Architectural test suite
└── README.md             # Project overview