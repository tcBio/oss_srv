"""
OSS_SRV Async Inference Client
Asynchronous client for concurrent requests
"""

import asyncio
from typing import List, Optional
from .client import InferenceClient, InferenceResult, InferenceConfig


class AsyncInferenceClient:
    """
    Asynchronous inference client for concurrent requests

    Example:
        >>> client = AsyncInferenceClient(model_path="model.engine")
        >>> result = await client.complete("Once upon a time")
        >>> print(result.text)
    """

    def __init__(self, model_path: str, executable_path: Optional[str] = None):
        """Initialize async inference client"""
        self.sync_client = InferenceClient(model_path, executable_path)
        self._executor = None

    async def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        config: Optional[InferenceConfig] = None
    ) -> InferenceResult:
        """
        Async generate completion for the given prompt

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            config: InferenceConfig object

        Returns:
            InferenceResult with generated text and metrics
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.sync_client.complete,
            prompt,
            max_tokens,
            temperature,
            top_p,
            config
        )

    async def batch_complete(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        config: Optional[InferenceConfig] = None,
        max_concurrent: int = 4
    ) -> List[InferenceResult]:
        """
        Async generate completions for multiple prompts concurrently

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per completion
            config: InferenceConfig object
            max_concurrent: Maximum concurrent requests

        Returns:
            List of InferenceResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(prompt):
            async with semaphore:
                return await self.complete(prompt, max_tokens=max_tokens, config=config)

        tasks = [process_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass
