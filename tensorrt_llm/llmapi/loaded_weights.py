from dataclasses import dataclass
from typing import Dict

import torch


# TODO: consider tradeoffs of having this class wrapper vs. using dict directly
@dataclass
class LoadedWeights:
    """GPU weight tensors extracted from a model for reuse.

    This allows creating a new LLM instance with different runtime configuration
    while reusing model weights already loaded in GPU memory, avoiding the cost
    of reloading weights from disk.

    Usage:
        llm = LLM(model="meta-llama/Llama-3.1-8B", max_batch_size=32)
        weights = llm.get_loaded_weights()
        llm.shutdown()
        llm2 = LLM(model="meta-llama/Llama-3.1-8B", max_batch_size=128, loaded_weights=weights)

    Note:
        The new LLM must have compatible configuration (same model, parallelism, dtype, etc.)
        for the weights to be reusable.
    """

    tensors: Dict[str, torch.Tensor]

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "LoadedWeights":
        """Extract weight tensor references from a model (no copy)."""
        return cls(tensors={name: param.data for name, param in model.named_parameters()})
