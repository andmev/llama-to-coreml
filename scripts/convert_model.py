import argparse
from pathlib import Path
from src.model import KvCacheStateLlamaForCausalLM
from src.converter import LlamaCoreMLConverter
from huggingface_hub import login

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

if __name__ == '__main__':
    main() 