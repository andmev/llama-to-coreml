# Llama 3.x CoreML Converter

This project converts the Llama 3.x model to CoreML format for deployment on Apple Silicon devices.


## Requirements

- Python 3.8+
- PyTorch 2.0+
- macOS 15+ (Sequoia)
- Apple Silicon Mac 

## Installation

```bash
mise install
```

[mise](https://github.com/jdx/mise) is a tool for managing Python versions and virtual environments.

## Usage

```bash
python scripts/convert_model.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --output-path output/llama_3.1_coreml.mlpackage \
    --token your_token_here
```

## Notes

- The model should be downloaded from HuggingFace.
- The token is optional, but it's required for gated models.
- This project based on this [article](https://machinelearning.apple.com/research/core-ml-on-device-llama)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
