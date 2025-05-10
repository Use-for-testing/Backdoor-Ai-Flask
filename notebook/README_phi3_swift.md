# Training Phi-3-mini-128k-instruct to Learn Swift Programming Language

This notebook trains Microsoft's Phi-3-mini-128k-instruct model to understand and work with Swift code using a dataset of real Swift files.

## Overview

The `phi3_swift_learning.ipynb` notebook adapts the original `idk.ipynb` notebook to use Microsoft's Phi-3-mini-128k-instruct model with a focus on learning the Swift programming language itself, not just classifying code.

## Key Features

1. **Language Learning Focus**: The model is trained to understand Swift syntax, patterns, and programming practices
2. **Diverse Instruction Prompts**: Uses a variety of prompt types to help the model learn different aspects of Swift:
   - Explaining code functionality
   - Identifying Swift language features
   - Completing or extending code
   - Suggesting improvements and best practices
   - Understanding code structure
   - Writing Swift functions
3. **Parameter-Efficient Fine-Tuning**: Uses LoRA to efficiently train the model
4. **Memory Optimization**: Implements 4-bit quantization to reduce memory requirements

## Dataset

The notebook uses the same dataset as the original notebook:
- Dataset ID: `mvasiliniuc/iva-swift-codeint`
- Contains real Swift code files from various categories (Models, Views, Controllers, etc.)
- The code is formatted into instruction-based prompts to help the model learn Swift

## Requirements

The notebook requires the following packages:
- transformers
- datasets
- torch
- scikit-learn
- tqdm
- accelerate
- peft (for LoRA fine-tuning)
- bitsandbytes (for quantization)

## Usage

1. Open the `phi3_swift_learning.ipynb` notebook in a Jupyter environment
2. Run all cells in sequence
3. The trained model will be saved to `./phi3_swift_model`

## Model Configuration

- Model: `microsoft/Phi-3-mini-128k-instruct`
- Max Sequence Length: 4096 tokens
- Batch Size: 4
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Number of Epochs: 3
- Gradient Accumulation Steps: 8
- LoRA Rank: 16
- LoRA Alpha: 32

## What the Model Learns

The model learns to:
1. **Understand Swift Syntax**: Recognizes and explains Swift language constructs
2. **Analyze Code**: Identifies patterns and features in Swift code
3. **Generate Code**: Completes or extends Swift code with appropriate functionality
4. **Apply Best Practices**: Suggests improvements and follows Swift conventions
5. **Explain Code**: Provides detailed explanations of Swift code functionality

## Testing

The notebook includes a testing section that evaluates the model on various Swift programming tasks:
- Explaining Swift syntax features
- Completing Swift functions
- Debugging Swift code
- Explaining Swift best practices

This helps verify that the model has truly learned the Swift language rather than just memorizing patterns.