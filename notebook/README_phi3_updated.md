# Training Phi-3-mini-128k-instruct on Swift Code with GPU and TPU Support

This notebook provides a comprehensive implementation for fine-tuning Microsoft's Phi-3-mini-128k-instruct model on Swift code, with optimized support for TPU, GPU, and CPU hardware.

## Key Features

### Hardware Acceleration Support
- **TPU Optimization**: Configured for efficient TPU training with bfloat16 precision and TPU-specific optimizations
- **GPU Optimization**: Implements 4-bit quantization and CUDA optimizations for efficient GPU training
- **CPU Fallback**: Gracefully falls back to CPU with appropriate optimizations when no accelerator is available

### Training Enhancements
- **Parameter-Efficient Fine-Tuning**: Uses LoRA with configurable parameters (r=16, alpha=32 by default)
- **Memory Optimization**: 
  - 4-bit quantization for GPU
  - bfloat16 for TPU
  - Gradient checkpointing
  - Efficient memory management
- **Detailed Monitoring**: Resource monitoring during training with hardware-specific metrics

### Swift Language Learning
- **Diverse Instruction Prompts**: Trains the model on various Swift programming tasks:
  - Explaining code functionality
  - Identifying language features
  - Completing or extending code
  - Suggesting improvements
  - Understanding code structure

## Hardware-Specific Configurations

### TPU Configuration
```python
# TPU-specific settings
use_fp16 = False  # TPUs work better with bf16 than fp16
use_bf16 = True   # Use bfloat16 for TPU
os.environ["XLA_USE_BF16"] = "1"  # Enable bfloat16 for better performance
```

### GPU Configuration
```python
# GPU-specific settings
use_fp16 = True   # Use fp16 for GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

### CPU Configuration
```python
# CPU-specific settings
use_fp16 = False  # No fp16 for CPU
use_bf16 = False  # No bf16 for CPU
# Reduced batch size and increased gradient accumulation
```

## Debugging Features

The notebook includes extensive debugging and monitoring capabilities:

1. **Hardware Detection**: Automatically detects and configures for available hardware
2. **Resource Monitoring**: Tracks memory usage, CPU/GPU utilization during training
3. **Detailed Logging**: Comprehensive logging with hardware-specific information
4. **Error Recovery**: Attempts to save checkpoints even when errors occur
5. **Memory Management**: Proactive memory cleanup to prevent OOM errors

## Requirements

The notebook requires the following packages with specific versions for compatibility:
```
transformers==4.38.2
datasets==2.16.1
evaluate==0.4.1
torch==2.1.2 (or torch==2.0.0 for TPU)
scikit-learn==1.4.0
tqdm==4.66.1
accelerate==0.27.2
peft==0.7.1
bitsandbytes==0.41.3
psutil==5.9.8
torch_xla[tpu]==2.0.0 (for TPU only)
```

## Usage

1. Open the notebook in a Jupyter environment with appropriate hardware (TPU, GPU, or CPU)
2. Run all cells in sequence
3. The notebook will automatically detect available hardware and configure accordingly
4. The trained model will be saved to `./phi3_swift_model`

## Model Evaluation

The notebook includes a testing section that evaluates the model on various Swift programming tasks:
- Explaining Swift syntax features
- Completing Swift functions
- Debugging Swift code
- Explaining Swift best practices

This helps verify that the model has truly learned the Swift language rather than just memorizing patterns.