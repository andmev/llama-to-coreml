from setuptools import setup, find_packages

setup(
    name="llama-coreml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers==4.44.2",
        "coremltools>=7.1",
        "numpy>=1.24.0",
        "sentencepiece",
        "protobuf",
    ],
) 