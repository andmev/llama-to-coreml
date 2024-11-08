import argparse
from pathlib import Path
import coremltools as ct
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import os
import json
import shutil
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTreeNode:
    def __init__(self, path: Path, is_file: bool = False):
        self.path = path
        self.is_file = is_file
        self.children = []
        self.is_weight_file = is_file and path.name.endswith('.bin')
        self.relative_path = None

    def __repr__(self):
        return f"{'File' if self.is_file else 'Dir'}: {self.path}"

def build_model_tree(root_path: Path, parent_path: Optional[Path] = None, is_root: bool = True) -> ModelTreeNode:
    """Recursively build a tree representation of the model package structure"""
    if is_root:
        # For root level, create nodes for its contents directly
        root_node = ModelTreeNode(root_path, is_file=False)
        root_node.relative_path = Path('')
        
        try:
            for item in root_path.iterdir():
                child_node = build_model_tree(item, root_path, is_root=False)
                root_node.children.append(child_node)
        except PermissionError as e:
            logger.warning(f"Permission denied accessing {root_path}: {e}")
        except Exception as e:
            logger.warning(f"Error accessing {root_path}: {e}")
            
        return root_node
    else:
        # For non-root nodes, process normally
        node = ModelTreeNode(root_path, is_file=root_path.is_file())
        node.relative_path = root_path.relative_to(parent_path)

        if root_path.is_dir():
            try:
                for item in root_path.iterdir():
                    child_node = build_model_tree(item, parent_path, is_root=False)
                    node.children.append(child_node)
            except PermissionError as e:
                logger.warning(f"Permission denied accessing {root_path}: {e}")
            except Exception as e:
                logger.warning(f"Error accessing {root_path}: {e}")

        return node

def find_weight_file(tree: ModelTreeNode) -> Optional[ModelTreeNode]:
    """Find the weights file in the model tree"""
    # For .mlmodelc files, weights are usually stored as 'weight.bin'
    # For .mlpackage files, they might be 'weights.bin'
    WEIGHT_FILE_NAMES = ['weights.bin', 'weight.bin']
    
    if tree.is_file and tree.path.name in WEIGHT_FILE_NAMES:
        return tree
    
    for child in tree.children:
        result = find_weight_file(child)
        if result:
            return result
    
    # If no specific weight file found, look for any large .bin file
    if tree.is_file and tree.path.suffix == '.bin':
        try:
            size = tree.path.stat().st_size
            # Only consider .bin files larger than 1MB as potential weight files
            if size > 1_000_000:  # 1MB
                logger.info(f"Found potential weight file: {tree.path} ({size / (1024*1024*1024):.2f} GB)")
                return tree
        except Exception as e:
            logger.warning(f"Error checking file size for {tree.path}: {e}")
    
    return None

def copy_tree_structure(
    tree: ModelTreeNode,
    src_root: Path,
    dst_root: Path,
    weight_file: Optional[ModelTreeNode] = None,
    chunk_start: int = 0,
    chunk_end: int = 0
) -> None:
    """Recursively copy the tree structure, handling only weights.bin specially"""
    dst_path = dst_root / tree.relative_path

    if tree.is_file:
        if tree == weight_file:  # This is weights.bin
            logger.info(f"Splitting weights.bin file: {tree.path}")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(tree.path, 'rb') as src, open(dst_path, 'wb') as dst:
                # Copy header
                header = src.read(4)
                dst.write(header)
                
                # Calculate and seek to chunk start
                src.seek(chunk_start + 4)  # +4 to skip header
                
                # Copy chunk data
                dst.write(src.read(chunk_end - chunk_start))
        else:  # Copy all other files exactly as they are
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tree.path, dst_path)
    else:
        # Create directory
        dst_path.mkdir(parents=True, exist_ok=True)
        
        # Process children
        for child in tree.children:
            copy_tree_structure(
                child,
                src_root,
                dst_root,
                weight_file,
                chunk_start,
                chunk_end
            )

def get_model_size(model_path: str) -> int:
    """Get the actual size of the model file/directory"""
    path = Path(model_path)
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return path.stat().st_size

def get_base_model_name(model_path: str) -> Tuple[str, str]:
    """
    Extract base model name and extension from path
    Returns tuple of (base_name, extension)
    """
    path = Path(model_path)
    extension = path.suffix  # .mlpackage or .mlmodelc
    base_name = path.stem   # name without extension
    return base_name, extension

def split_mlpackage(model_path: str, num_chunks: int, output_dir: str):
    """
    Split a large CoreML model into chunks
    
    Args:
        model_path: Path to the CoreML model (.mlpackage or .mlmodelc)
        num_chunks: Number of chunks to split the model into
        output_dir: Directory to save the split models
    """
    logger.info(f"Loading model from {model_path}")
    model_path = Path(model_path)
    
    # Build tree representation of the model package
    logger.info("Building model tree structure...")
    model_tree = build_model_tree(model_path)
    
    # Find the weights file
    weight_file = find_weight_file(model_tree)
    if not weight_file:
        raise ValueError("No weight file found in the model package")
    
    logger.info(f"Found weight file: {weight_file.path}")
    
    # Get weight file size
    weight_size = weight_file.path.stat().st_size
    logger.info(f"Weight file size: {weight_size / (1024*1024*1024):.2f} GB")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get base model name and extension
    base_model_name, extension = get_base_model_name(str(model_path))
    
    # Create chunks
    for i in range(num_chunks):
        # Format chunk name based on original extension
        chunk_name = f"{base_model_name}-chunk{i+1}{extension}"
        chunk_path = output_path / chunk_name
        
        logger.info(f"Creating chunk {i+1}: {chunk_path}")
        
        # Calculate chunk boundaries
        if i == 0:  # First chunk
            chunk_start = 0
            chunk_end = int(weight_size / num_chunks)
        elif i == num_chunks - 1:  # Last chunk
            chunk_start = int((num_chunks - 1) * weight_size / num_chunks)
            chunk_end = weight_size
        else:  # Middle chunks
            chunk_start = int(i * weight_size / num_chunks)
            chunk_end = int((i + 1) * weight_size / num_chunks)
        
        # Copy tree structure with split weights
        copy_tree_structure(
            model_tree,
            model_path,
            chunk_path,
            weight_file,
            chunk_start,
            chunk_end
        )
        
        # Update metadata if it exists
        metadata_files = list(chunk_path.rglob("metadata.json"))
        for metadata_path in metadata_files:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Update metadata for this chunk
                if 'inputSchema' in metadata:
                    for input_desc in metadata['inputSchema']:
                        input_desc['name'] = f"{chunk_name}_{input_desc['name']}"
                
                if 'outputSchema' in metadata:
                    for output_desc in metadata['outputSchema']:
                        output_desc['name'] = f"{chunk_name}_{output_desc['name']}"
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        chunk_size = get_model_size(str(chunk_path))
        logger.info(f"Chunk {i+1} size: {chunk_size / (1024*1024*1024):.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='Split CoreML model into chunks')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the CoreML model to split')
    parser.add_argument('--num-chunks', type=int, required=True,
                      help='Number of chunks to split the model into')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save the split models')
    
    args = parser.parse_args()
    split_mlpackage(args.model_path, args.num_chunks, args.output_dir)

if __name__ == '__main__':
    main() 