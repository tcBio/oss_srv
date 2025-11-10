"""
OSS_SRV Python SDK
High-performance inference client for OSS-20B on Blackwell GPUs

Usage:
    from oss_srv import InferenceClient

    client = InferenceClient(model_path="model.engine")
    result = client.complete("Once upon a time", max_tokens=100)
    print(result.text)
"""

from .client import InferenceClient, InferenceResult, InferenceConfig
from .async_client import AsyncInferenceClient
from .batch_client import BatchInferenceClient

__version__ = "1.0.0"
__all__ = [
    "InferenceClient",
    "InferenceResult",
    "InferenceConfig",
    "AsyncInferenceClient",
    "BatchInferenceClient",
]
