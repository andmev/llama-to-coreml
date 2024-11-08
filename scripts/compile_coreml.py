import argparse
import os
import subprocess
from pathlib import Path

def compile_model(input_path: str, output_dir: str):
    """
    Compile .mlpackage model to .mlmodelc format using coremlcompiler
    
    Args:
        input_path: Path to the .mlpackage model
        output_dir: Directory to save the compiled model
    """
    try:
        # Ensure input path exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input model not found at: {input_path}")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare output path
        output_path = os.path.join(output_dir, Path(input_path).stem + '.mlmodelc')
        
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
    parser = argparse.ArgumentParser(description='Compile Core ML model package to .mlmodelc format')
    parser.add_argument('input_path', type=str, help='Path to the .mlpackage model')
    parser.add_argument('output_dir', type=str, help='Directory to save the compiled model')
    
    args = parser.parse_args()
    compile_model(args.input_path, args.output_dir)

if __name__ == '__main__':
    main() 