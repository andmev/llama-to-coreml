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

This tool provides two main scripts for working with CoreML models:

### Converting Models

To convert a Llama model to CoreML format:

```shell
python scripts/convert_model.py \
    --model-path meta-llama/Llama-3.2-3B-Instruct \
    --output-path output/Llama-3.2-3B-Instruct.mlpackage \
    --token your_token_here \
    --compile  # Optional: add this flag to compile after conversion
```

The `--compile` flag will automatically compile the model to `.mlmodelc` format after conversion, creating it in the same directory as the `.mlpackage` file.

### Compiling Models

To compile a CoreML model to `.mlmodelc` format:

```shell
python scripts/compile_model.py \
    --model-path output/Llama-3.2-3B-Instruct.mlpackage \
    --output-path output/Llama-3.2-3B-Instruct.mlmodelc
```

## Notes

- The model should be downloaded from HuggingFace.
- The token is optional, but it's required for gated models.
- This project based on this [article](https://machinelearning.apple.com/research/core-ml-on-device-llama)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

