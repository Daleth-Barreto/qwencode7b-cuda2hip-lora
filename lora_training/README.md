# LoRA Training Module

This directory contains the core training logic for the **Qwen-CUDA2HIP** project.

## Structure
- `Training.ipynb`: The complete pipeline from data ingestion to Hugging Face Hub deployment.
- `qwen_cuda2hip_lora/`: (Generated after training) Contains the `adapter_model.bin` and configuration files.

## Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Base Model** | Qwen2.5-Coder-7B-Instruct |
| **Optimizer** | AdamW (Torch) |
| **Precision** | BF16 |
| **LoRA R** | 16 |
| **LoRA Alpha** | 32 |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 1 (Acc. Steps: 8) |

## Training Performance
The model achieved a significant reduction in loss (from ~0.46 to ~0.31) over 2 epochs, indicating a successful adaptation to the HIP syntax and reasoning patterns.