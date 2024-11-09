import argparse
import subprocess
from pathlib import Path

def compile_model(input_path: str, output_path: str = None):
    """
    Compile .mlpackage model to .mlmodelc format using coremlcompiler
    
    Args:
        input_path: Path to the .mlpackage model
        output_path: Optional custom path to save the compiled model. 
                    If not provided, will use same directory as input_path
    """
    try:
        if output_path is None:
            # Use the same directory as input_path
            output_path = str(Path(input_path).parent)
        
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
    parser = argparse.ArgumentParser(description='Compile CoreML model to mlmodelc format')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the .mlpackage model to compile')
    parser.add_argument('--output-path', type=str,
                      help='Custom path to save the compiled model (optional)')
    
    args = parser.parse_args()
    compile_model(args.model_path, args.output_path)

if __name__ == '__main__':
    main() 