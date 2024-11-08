import argparse
import os
import subprocess
from pathlib import Path
from src.model import KvCacheStateLlamaForCausalLM
from src.converter import LlamaCoreMLConverter
from huggingface_hub import login

def compile_model(input_path: str):
    """
    Compile .mlpackage model to .mlmodelc format using coremlcompiler
    
    Args:
        input_path: Path to the .mlpackage model
    """
    try:
        output_path = str(Path(input_path).with_suffix('.mlmodelc'))
        
        # Use coremlcompiler to compile the model
        cmd = ['xcrun', 'coremlcompiler', 'compile', input_path, output_path]
        
        print(f"Running compilation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Model successfully compiled to: {output_path}")
        else:
            print(f"Compilation failed with error:\n{result.stderr}")
            raise Exception(result.stderr)
            
    except Exception as e:
        print(f"Error during compilation: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert Llama 3.x to CoreML')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the Llama 3.x model')
    parser.add_argument('--output-path', type=str, required=True,
                      help='Path to save the CoreML model')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for the model')
    parser.add_argument('--context-size', type=int, default=2048,
                      help='Context size for the model')
    parser.add_argument('--no-quantize', action='store_true',
                      help='Disable Int4 quantization')
    parser.add_argument('--token', type=str,
                      help='HuggingFace token for downloading models')
    parser.add_argument('--compile', action='store_true',
                      help='Compile the model after conversion')
    
    args = parser.parse_args()

    # Login to HuggingFace if token is provided
    if args.token:
        login(args.token)

    # Load the model
    model = KvCacheStateLlamaForCausalLM(
        args.model_path,
        batch_size=args.batch_size,
        context_size=args.context_size
    )

    # Convert to CoreML
    converter = LlamaCoreMLConverter(
        model,
        batch_size=args.batch_size,
        context_size=args.context_size
    )
    
    mlmodel = converter.convert(quantize=not args.no_quantize)

    # Save the model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    
    # Compile if requested
    if args.compile:
        compile_model(str(output_path))

if __name__ == '__main__':
    main() 