# Llama 3.x CoreML Converter

This project converts the Llama 3.x model to CoreML format for deployment on Apple Silicon devices.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- macOS 15+ (Sequoia)
- Apple Silicon Mac 

## Installation

```shell
mise install
```

[mise](https://github.com/jdx/mise) is a tool for managing Python versions and virtual environments.

## Model Conversion and Compilation

This tool provides scripts for converting, compiling, and splitting CoreML models:

### Converting Models

To convert a Llama model to CoreML format:

```shell
python scripts/convert_model.py \
    --model-path meta-llama/Llama-3.2-3B-Instruct \
    --output-path output/Llama-3.2-3B-Instruct.mlpackage \
    --token your_token_here \
    --compression quantization \
    --split-chunks 2 \
    --compile  # Optional: add this flag to compile after conversion
```

#### Compression Options

The converter supports several compression types to reduce model size:
- `none`: No compression
- `palettization`: Uses palette-based compression
- `quantization`: Uses weight quantization (default)
- `mixed`: Combines different compression techniques

#### Model Splitting

Use `--split-chunks` to split the model into multiple smaller files:
- Set the number of chunks (e.g., `--split-chunks 2` for two parts)
- Split models will be saved in a `split_models` directory
- Useful for handling large models or memory constraints

### Compiling Models

To compile a CoreML model to `.mlmodelc` format:

```shell
python scripts/compile_model.py \
    --model-path output/Llama-3.2-3B-Instruct.mlpackage \
    --output-path output/Llama-3.2-3B-Instruct.mlmodelc
```

### Splitting Existing Models

To split an existing CoreML model into chunks:

```shell
python scripts/split_model.py \
    --model-path output/Llama-3.2-3B-Instruct.mlpackage \
    --num-chunks 2 \
    --output-dir output/split_models
```

## Notes

- The model should be downloaded from HuggingFace
- The token is optional, but it's required for gated models
- Split models can be used to distribute model loading across multiple processes
- This project is based on this [article](https://machinelearning.apple.com/research/core-ml-on-device-llama).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

