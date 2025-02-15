# PEFT Adapters Project

This project demonstrates fine-tuning Large Language Models (LLMs) using LoRA adapters, leveraging the Unsloth library for optimized training, it has been modularized based on the unsloth notebook. 

## Overview

The project includes functionality for:
- Loading and preparing base models
- Adding LoRA adapters
- Training models on custom datasets
- Performing inference with trained models
- Generating training datasets

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth
- Datasets
- TRL

## Project Structure

```
peft-adapters/
├── utils/
│   ├── __init__.py
│   ├── constants.py      # Project-wide constants
│   ├── load_models.py    # Model loading utilities
│   ├── training.py       # Training functionality
│   └── inference.py      # Inference utilities
├── datasets/             # Directory for training datasets
├── training_with_unsloth.py    # Main training script
├── dataset-generator-numbers.py # Dataset generation script
└── README.md
```

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install torch transformers unsloth datasets trl
```

## Usage

### Generate Training Dataset
```bash
python dataset-generator-numbers.py
```

### Train Model
```bash
python training_with_unsloth.py
```

### Key Features

- **Efficient Training**: Uses Unsloth's optimizations for faster training
- **4-bit Quantization**: Supports loading models in 4-bit precision
- **Custom Dataset Generation**: Includes tools for creating training datasets
- **Streaming Inference**: Supports both regular and streaming inference modes

## Model Configuration

The project uses the following default settings:
- Maximum sequence length: 2048
- LoRA rank: 8
- Batch size: 2
- Learning rate: 2e-4

These can be modified in the respective configuration files.

## Contributing

Feel free to open issues or submit pull requests for improvements, for more information check my [YouTube Channel](https://www.youtube.com/@AlexOtano)

## License

MIT