import argparse
import coremltools as ct
from pathlib import Path

# TODO: Wait until coremltools issue #2353 is resolved https://github.com/apple/coremltools/issues/2353
def bisect_model(model_path: str, output_path: str = None, merge_chunks: bool = False):
    if output_path is None:
        # Use the same directory as input_path
        output_path = str(Path(model_path).parent)

    # The following code will produce two smaller models:
    # `./output/my_model_chunk1.mlpackage` and `./output/my_model_chunk2.mlpackage`
    # It also compares the output numerical of the original Core ML model with the chunked models.
    ct.models.utils.bisect_model(
        model_path,
        output_path,
        merge_chunks_to_pipeline=merge_chunks,
    )

def main():
    parser = argparse.ArgumentParser(description='Bisect CoreML model')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the .mlpackage model to bisect')
    parser.add_argument('--output-path', type=str,
                      help='Custom path to save the bisected model (optional)')
    parser.add_argument('--merge-chunks', action='store_true',
                      help='Merge the chunks into a single pipeline model')
    
    args = parser.parse_args()
    bisect_model(args.model_path, args.output_path, args.merge_chunks)

if __name__ == '__main__':
    main() 