import coremltools as ct
import torch
import numpy as np
from enum import Enum

class CompressionType(Enum):
    NONE = "none"
    PALETTIZATION = "palettization" 
    QUANTIZATION = "quantization"
    MIXED = "mixed"

class LlamaCoreMLConverter:
    def __init__(
        self,
        model,
        batch_size: int = 1,
        context_size: int = 2048
    ):
        self.model = model
        self.batch_size = batch_size
        self.context_size = context_size

    def convert(self, compression_type: CompressionType = CompressionType.QUANTIZATION) -> ct.models.MLModel:
        # Initialize and trace PyTorch model
        example_inputs = (
            torch.zeros((1, 2), dtype=torch.int32),
            torch.zeros((1, 1, 2, 5), dtype=torch.float32)
        )
        traced_model = torch.jit.trace(
            self.model.eval(),
            example_inputs=example_inputs
        )

        # Convert to Core ML
        query_size = ct.RangeDim(lower_bound=1, upper_bound=self.context_size, default=1)
        final_step = ct.RangeDim(lower_bound=1, upper_bound=self.context_size, default=1)
        
        inputs = [
            ct.TensorType(
                shape=(self.batch_size, query_size),
                dtype=np.int32,
                name="inputIds"
            ),
            ct.TensorType(
                shape=(self.batch_size, 1, query_size, final_step),
                dtype=np.float16,
                name="causalMask"
            ),
        ]
        
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=self.model.kv_cache_shape,
                    dtype=np.float16
                ),
                name="keyCache"
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=self.model.kv_cache_shape,
                    dtype=np.float16
                ),
                name="valueCache"
            ),
        ]

        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=[ct.TensorType(dtype=np.float16, name="logits")],
            states=states,
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        )

        if compression_type != CompressionType.NONE:
            mlmodel = self._compress_model(mlmodel, compression_type)

        return mlmodel

    def _compress_model(self, mlmodel: ct.models.MLModel, compression_type: CompressionType) -> ct.models.MLModel:
        if compression_type == CompressionType.PALETTIZATION:
            # Use palettization for better compression
            op_config = ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_grouped_channel",
                group_size=16
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            return ct.optimize.coreml.palettize_weights(mlmodel, config)

        elif compression_type == CompressionType.QUANTIZATION:
            # Use linear quantization
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_block",
                block_size=32
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            return ct.optimize.coreml.linear_quantize_weights(mlmodel, config)

        elif compression_type == CompressionType.MIXED:
            # Apply both activation and weight quantization
            # First quantize activations
            act_config = ct.optimize.coreml.OpActivationLinearQuantizerConfig(
                mode="linear_symmetric"
            )
            act_model_config = ct.optimize.coreml.OptimizationConfig(global_config=act_config)
            mlmodel = ct.optimize.coreml.linear_quantize_activations(
                mlmodel, 
                act_model_config
            )

            # Then quantize weights
            weight_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8"
            )
            weight_model_config = ct.optimize.coreml.OptimizationConfig(weight_config)
            return ct.optimize.coreml.linear_quantize_weights(
                mlmodel,
                weight_model_config
            ) 