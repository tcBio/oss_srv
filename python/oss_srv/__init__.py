"""
OSS_SRV Python SDK
High-performance inference client for OSS-20B on Blackwell GPUs

Usage:
    from oss_srv import InferenceClient

    client = InferenceClient(model_path="model.engine")
    result = client.complete("Once upon a time", max_tokens=100)
    print(result.text)

Streaming Usage:
    from oss_srv import StreamingInferenceClient

    client = StreamingInferenceClient(model_path="model.engine")
    for token in client.stream("Once upon a time"):
        print(token.text, end='', flush=True)
"""

from .client import InferenceClient, InferenceResult, InferenceConfig
from .async_client import AsyncInferenceClient
from .batch_client import BatchInferenceClient
from .streaming_client import (
    StreamingInferenceClient,
    StreamingToken,
    SSEStreamingClient,
    WebSocketStreamingClient
)

__version__ = "1.0.0"
__all__ = [
    "InferenceClient",
    "InferenceResult",
    "InferenceConfig",
    "AsyncInferenceClient",
    "BatchInferenceClient",
    "StreamingInferenceClient",
    "StreamingToken",
    "SSEStreamingClient",
    "WebSocketStreamingClient",
]
