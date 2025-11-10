"""
OSS_SRV Batch Inference Client
Optimized client for batch processing
"""

from typing import List, Optional, Iterator
from dataclasses import dataclass
from .client import InferenceClient, InferenceResult, InferenceConfig


@dataclass
class BatchRequest:
    """Single request in a batch"""
    id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class BatchResult:
    """Result from batch processing"""
    id: str
    result: InferenceResult


class BatchInferenceClient:
    """
    Batch inference client with optimized processing

    Example:
        >>> client = BatchInferenceClient(model_path="model.engine", batch_size=8)
        >>> requests = [
        ...     BatchRequest(id="1", prompt="Once upon a time"),
        ...     BatchRequest(id="2", prompt="In a galaxy far away"),
        ... ]
        >>> results = client.process_batch(requests)
        >>> for result in results:
        ...     print(f"{result.id}: {result.result.text}")
    """

    def __init__(
        self,
        model_path: str,
        executable_path: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Initialize batch inference client

        Args:
            model_path: Path to TensorRT engine
            executable_path: Path to executable
            batch_size: Maximum batch size for processing
        """
        self.client = InferenceClient(model_path, executable_path)
        self.batch_size = batch_size

    def process_batch(
        self,
        requests: List[BatchRequest],
        config: Optional[InferenceConfig] = None
    ) -> List[BatchResult]:
        """
        Process a batch of requests

        Args:
            requests: List of BatchRequest objects
            config: Optional InferenceConfig

        Returns:
            List of BatchResult objects
        """
        results = []

        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]

            for req in batch:
                # Process request
                result = self.client.complete(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    config=config
                )

                results.append(BatchResult(id=req.id, result=result))

        return results

    def stream_batch(
        self,
        requests: List[BatchRequest],
        config: Optional[InferenceConfig] = None
    ) -> Iterator[BatchResult]:
        """
        Stream batch results as they complete

        Args:
            requests: List of BatchRequest objects
            config: Optional InferenceConfig

        Yields:
            BatchResult objects as they complete
        """
        for req in requests:
            result = self.client.complete(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                config=config
            )

            yield BatchResult(id=req.id, result=result)
