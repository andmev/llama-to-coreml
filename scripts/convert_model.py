import argparse
from pathlib import Path
from src.model import KvCacheStateLlamaForCausalLM
from src.converter import LlamaCoreMLConverter, CompressionType
from huggingface_hub import login
from scripts.compile_model import compile_model
from scripts.split_model import split_model

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
    parser.add_argument('--compression', type=str, choices=['none', 'palettization', 'quantization', 'mixed'],
                      default='quantization', help='Compression type to use')
    parser.add_argument('--split-chunks', type=int, default=0,
                      help='Number of chunks to split the model into (0 for no splitting)')
    parser.add_argument('--token', type=str,
                      help='HuggingFace token for downloading models')
    parser.add_argument('--compile', action='store_true',
                      help='Compile the model after conversion')
    
    args = parser.parse_args()

    if args.token:
        login(args.token)

    # Load and convert model
    model = KvCacheStateLlamaForCausalLM(
        args.model_path,
        batch_size=args.batch_size,
        context_size=args.context_size
    )

    converter = LlamaCoreMLConverter(
        model,
        batch_size=args.batch_size,
        context_size=args.context_size
    )
    
    compression_type = CompressionType(args.compression)
    mlmodel = converter.convert(compression_type=compression_type)

    # Save the model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    
    # Split model if requested
    if args.split_chunks > 0:
        split_dir = output_path.parent / "split_models"
        split_model(str(output_path), args.split_chunks, str(split_dir))
    
    # Compile if requested
    if args.compile:
        compile_path = str(output_path.with_suffix('.mlmodelc'))
        compile_model(str(output_path), compile_path)

if __name__ == '__main__':
    main() 